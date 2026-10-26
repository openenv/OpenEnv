import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# MoE Gated GEMM (Mixture of Experts with Fused Gating)
# Used in: Mixtral, DeepSeek-V3, Grok, DBRX, Arctic
# Reference: https://arxiv.org/abs/2401.04088 (Mixtral of Experts)
#
# This problem focuses on the "gated dual GEMM" pattern in MoE FFNs:
#   output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
#
# The baseline uses batched matrix multiplication to process all experts
# in parallel (no sequential loop). A custom CUDA kernel should:
# 1. Fuse gate_proj and up_proj into single memory read of x
# 2. Fuse SiLU activation with the elementwise multiply
# 3. Use grouped GEMM for better utilization with varying expert batch sizes
# 4. Optimize the gather/scatter pattern for expert weight selection
# 5. Target 2-3x speedup through fusion and memory optimization


class Model(nn.Module):
    """
    MoE Expert with Gated GEMM (SiLU-gated FFN).

    This is a SINGLE expert's computation pattern, used in MoE FFN:
    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    The "gated GEMM" refers to: SiLU(gate_proj(x)) * up_proj(x)
    This is two parallel GEMMs followed by element-wise multiply.

    Key optimization targets:
    1. Fuse gate_proj and up_proj into single memory read of x
    2. Fuse SiLU activation with multiplication
    3. Optimize memory layout for the dual GEMM pattern
    4. When batched across experts, enable parallel execution

    The naive implementation runs two separate matmuls.
    An optimized kernel should read x once and compute both projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # Expert weights: each expert has gate_proj, up_proj, down_proj
        # Shape: (num_experts, out_features, in_features) for batched matmul
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, hidden_size)
        expert_indices: torch.Tensor,  # (batch, seq_len, top_k) - selected expert indices
        expert_weights: torch.Tensor,  # (batch, seq_len, top_k) - routing weights
    ) -> torch.Tensor:
        """
        MoE forward with gated dual GEMM.

        Each token is processed by top_k experts, weighted by expert_weights.
        This implementation groups tokens by expert and uses efficient batched
        operations. The expert loop uses torch operations that can be compiled.

        Optimization target: A CUDA kernel should:
        1. Fuse gate_proj and up_proj into single memory read of x
        2. Fuse SiLU with the elementwise multiply
        3. Use grouped GEMM (CUTLASS) for varying expert batch sizes
        4. Avoid the explicit sort/gather/scatter overhead
        5. Target 2-3x speedup through fusion
        """
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        num_tokens = batch * seq_len

        x_flat = x.view(num_tokens, self.hidden_size)
        indices_flat = expert_indices.view(num_tokens * top_k)
        weights_flat = expert_weights.view(num_tokens * top_k)

        # Create token indices for each (token, slot) pair
        token_ids = torch.arange(num_tokens, device=x.device)
        token_ids = token_ids.unsqueeze(1).expand(-1, top_k).reshape(-1)

        # Sort by expert to enable batched processing
        sorted_expert_idx, sort_order = indices_flat.sort()
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = weights_flat[sort_order]

        # Get expert boundaries
        expert_counts = torch.bincount(sorted_expert_idx, minlength=self.num_experts)
        expert_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=x.device), expert_counts.cumsum(0)]
        )

        # Gather sorted inputs
        sorted_x = x_flat[sorted_token_ids]  # (N*top_k, H)

        # Process all experts - vectorized within each expert group
        sorted_output = torch.empty_like(sorted_x)

        for e in range(self.num_experts):
            start, end = expert_offsets[e].item(), expert_offsets[e + 1].item()
            if start == end:
                continue

            expert_x = sorted_x[start:end]  # (n_e, H)

            # Gated dual GEMM for this expert
            gate = F.silu(F.linear(expert_x, self.gate_proj[e]))
            up = F.linear(expert_x, self.up_proj[e])
            intermediate = gate * up
            sorted_output[start:end] = F.linear(intermediate, self.down_proj[e])

        # Apply weights and scatter back
        weighted_sorted = sorted_output * sorted_weights.unsqueeze(-1)

        # Scatter-add back to original token positions
        output = torch.zeros(
            num_tokens, self.hidden_size, device=x.device, dtype=x.dtype
        )
        output.index_add_(0, sorted_token_ids, weighted_sorted)

        return output.view(batch, seq_len, self.hidden_size)


# Mixtral-style configuration
batch_size = 4
seq_len = 2048
hidden_size = 4096
intermediate_size = 14336  # Mixtral uses large intermediate
num_experts = 8
top_k = 2  # Each token routed to 2 experts


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Random expert selection (in real MoE, this comes from gating network)
    expert_indices = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(batch_size * seq_len)]
    ).view(batch_size, seq_len, top_k)

    # Random routing weights (normalized)
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)

    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_experts]
