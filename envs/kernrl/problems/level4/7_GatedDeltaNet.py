import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops import chunk_gated_delta_rule

# Gated DeltaNet: Linear Attention with Gated Delta Rule
# Reference: https://arxiv.org/abs/2412.06464 (ICLR 2025)
#
# Core recurrence:
#   S_t = alpha_t * S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
#   o_t = S_t @ q_t
#
# This baseline uses flash-linear-attention's chunk-wise parallel algorithm.
# The chunked approach uses the WY representation to parallelize across
# sequence length, achieving near-optimal hardware utilization.
#
# A custom CUDA kernel would need to match or beat fla's Triton implementation:
# 1. Chunk-wise parallel processing with WY representation
# 2. Fused operations within each chunk
# 3. Efficient inter-chunk state propagation
# 4. Memory-efficient gradient computation (if training)
# 5. Target: match fla performance or achieve 1.2-1.5x through custom fusion


def gated_delta_attention(
    q: torch.Tensor,  # (batch, heads, seq, d_qk)
    k: torch.Tensor,  # (batch, heads, seq, d_qk)
    v: torch.Tensor,  # (batch, heads, seq, d_v)
    alpha: torch.Tensor,  # (batch, heads, seq) - decay gate (0-1)
    beta: torch.Tensor,  # (batch, heads, seq) - update gate (0-1)
    scale: float,
) -> torch.Tensor:
    """
    Gated delta rule attention using flash-linear-attention's optimized kernel.

    The fla library implements chunk-wise parallelization with the WY
    representation, enabling efficient GPU utilization. This is the
    state-of-the-art implementation for this recurrence.
    """
    # fla expects gate in log-space for numerical stability
    g = alpha.clamp(min=1e-6).log()

    # chunk_gated_delta_rule returns (output, final_state)
    output, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)
    return output


class Model(nn.Module):
    """
    Gated DeltaNet: Linear Attention with Gated Delta Rule

    This baseline uses flash-linear-attention's optimized Triton kernels
    which implement chunk-wise parallelization with the WY representation.
    A custom CUDA kernel should match or beat fla's throughput.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_short_conv = use_short_conv

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        if use_short_conv:
            self.q_conv = nn.Conv1d(
                num_heads * head_dim_qk,
                num_heads * head_dim_qk,
                kernel_size=conv_kernel_size,
                groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1,
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk,
                num_heads * head_dim_qk,
                kernel_size=conv_kernel_size,
                groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1,
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v,
                num_heads * head_dim_v,
                kernel_size=conv_kernel_size,
                groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1,
            )

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)
        self.scale = head_dim_qk**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape to (B, H, T, D) for recurrence
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).transpose(1, 2)

        alpha = torch.sigmoid(self.a_proj(x)).transpose(1, 2)  # (B, H, T)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)

        # Chunk-wise parallel attention (fla)
        o = gated_delta_attention(q, k, v, alpha, beta, scale=self.scale)

        # (B, H, T, d_v) -> (B, T, H, d_v)
        o = o.transpose(1, 2)

        o = self.o_norm(o)

        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128
head_dim_v = 128


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
