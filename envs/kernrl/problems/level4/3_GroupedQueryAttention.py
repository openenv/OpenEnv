import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Grouped Query Attention (GQA)
# Used in: Llama 2 70B, Mistral, Llama 3, Gemma, Qwen 2.5, etc.
# Reference: https://arxiv.org/abs/2305.13245 (GQA: Training Generalized Multi-Query Transformer)
#
# GQA is a memory-efficient attention variant where multiple query heads share
# the same key/value heads. This reduces KV cache size while maintaining quality.
#
# Standard MHA: n_heads query heads, n_heads KV heads (ratio 1:1)
# MQA: n_heads query heads, 1 KV head (all queries share same KV)
# GQA: n_heads query heads, n_kv_heads KV heads (n_heads // n_kv_heads queries per KV)
#
# Optimization targets:
# 1. KV head broadcasting/expansion to query heads
# 2. Fused attention with grouped structure
# 3. Memory layout optimization for KV cache access patterns


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


class Model(nn.Module):
    """
    Grouped Query Attention (GQA)

    Key optimization targets:
    1. Efficient KV head expansion/repeat to match query heads
    2. Fused QKV projection with grouped structure
    3. Memory-efficient attention with reduced KV heads
    4. RoPE application fused with attention

    The naive implementation repeats KV heads to match query heads.
    An optimized kernel should:
    - Avoid explicit KV expansion (compute attention with implicit repeat)
    - Fuse RoPE with attention computation
    - Optimize memory access patterns for grouped structure
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.softmax_scale = head_dim ** (-0.5)

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads to match query heads.
        This is the INEFFICIENT operation that should be avoided in fused kernel.

        Input: (batch, num_kv_heads, seq_len, head_dim)
        Output: (batch, num_attention_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # INEFFICIENT: Expand KV heads to match query heads
        # This is the main optimization target - avoid explicit memory expansion
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


# Llama 3 70B style configuration (scaled down for single H100)
# Full Llama 3 70B: 64 query heads, 8 KV heads (8:1 ratio)
batch_size = 4
seq_len = 2048
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8  # 4:1 grouping ratio
head_dim = 128
max_position_embeddings = 4096


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
    ]
