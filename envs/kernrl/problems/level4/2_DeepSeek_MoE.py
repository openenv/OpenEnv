import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# DeepSeek-V3 Mixture of Experts (MoE) Layer
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Reference: https://arxiv.org/abs/2412.19437 (DeepSeek-V3 Technical Report)
#
# This implements the MoE layer with:
# - Auxiliary-free load balancing via bias correction (noaux_tc gating)
# - Grouped expert selection (n_group groups, topk_group groups selected)
# - Shared experts processed in parallel with routed experts
#
# The baseline uses batched expert computation with stacked weights.
# A fused CUDA kernel can further optimize memory access patterns.


class MoEGate(nn.Module):
    """
    DeepSeek-V3 MoE gating with grouped expert selection.

    Uses sigmoid scoring and selects top-k experts from top-k groups.
    Bias correction (e_score_correction_bias) enables auxiliary-free load balancing.
    Note: Grouped selection is inference-only; bias is learned during training.
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        # Bias is a buffer, not a parameter - updated via load statistics, not gradients
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        # Compute gating scores with sigmoid (not softmax like standard MoE)
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()

        # Apply bias correction for load balancing
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Grouped selection: select top-k groups, then top-k experts within those groups
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask out experts not in selected groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        # Get weights for selected experts
        topk_weight = scores.gather(1, topk_idx)

        # Normalize weights
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class Model(nn.Module):
    """
    DeepSeek-V3 Mixture of Experts Layer

    Uses batched expert computation with stacked weights for efficient parallel execution.
    All expert weights are stored in single tensors: (n_experts, out_features, in_features)

    Key optimization targets for CUDA kernel:
    1. Fused gather + batched GEMM for expert computation
    2. Memory-efficient token-to-expert routing
    3. Coalesced memory access patterns for stacked weights
    4. Fused weighted scatter-add for output combination
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        n_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts

        # Stacked expert weights for batched computation
        # Shape: (n_experts, out_features, in_features)
        self.gate_proj = nn.Parameter(
            torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_routed_experts, hidden_size, intermediate_size) * 0.02
        )

        # Gating network
        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
        )

        # Optional shared experts (processed for all tokens)
        if n_shared_experts > 0:
            shared_intermediate = intermediate_size * n_shared_experts
            self.shared_gate_proj = nn.Linear(
                hidden_size, shared_intermediate, bias=False
            )
            self.shared_up_proj = nn.Linear(
                hidden_size, shared_intermediate, bias=False
            )
            self.shared_down_proj = nn.Linear(
                shared_intermediate, hidden_size, bias=False
            )
        else:
            self.shared_gate_proj = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert not self.training, "DeepSeek MoE grouped selection is inference-only"

        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, _ = orig_shape

        # Get expert routing
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]

        # Batched expert computation
        # topk_idx: (num_tokens, top_k) - which experts each token uses
        # topk_weight: (num_tokens, top_k) - routing weights

        # Flatten token-expert pairs
        # Each token is processed by top_k experts, so we have num_tokens * top_k computations
        flat_topk_idx = topk_idx.view(-1)  # (num_tokens * top_k,)

        # Expand tokens to match expert assignments
        # (num_tokens, hidden) -> (num_tokens, top_k, hidden) -> (num_tokens * top_k, hidden)
        expanded_tokens = hidden_states.unsqueeze(1).expand(
            -1, self.num_experts_per_tok, -1
        )
        expanded_tokens = expanded_tokens.reshape(
            -1, self.hidden_size
        )  # (num_tokens * top_k, hidden)

        # Gather expert weights for each token-expert pair
        # gate_proj[expert_idx]: (intermediate, hidden)
        selected_gate = self.gate_proj[
            flat_topk_idx
        ]  # (num_tokens * top_k, intermediate, hidden)
        selected_up = self.up_proj[
            flat_topk_idx
        ]  # (num_tokens * top_k, intermediate, hidden)
        selected_down = self.down_proj[
            flat_topk_idx
        ]  # (num_tokens * top_k, hidden, intermediate)

        # Batched expert MLP: down(silu(gate(x)) * up(x))
        # x: (num_tokens * top_k, hidden, 1)
        x = expanded_tokens.unsqueeze(-1)

        # gate(x): (num_tokens * top_k, intermediate, hidden) @ (num_tokens * top_k, hidden, 1)
        #        = (num_tokens * top_k, intermediate, 1)
        gate_out = torch.bmm(selected_gate, x).squeeze(
            -1
        )  # (num_tokens * top_k, intermediate)
        up_out = torch.bmm(selected_up, x).squeeze(
            -1
        )  # (num_tokens * top_k, intermediate)

        # SiLU activation and element-wise multiply
        intermediate = F.silu(gate_out) * up_out  # (num_tokens * top_k, intermediate)

        # down projection
        expert_out = torch.bmm(selected_down, intermediate.unsqueeze(-1)).squeeze(
            -1
        )  # (num_tokens * top_k, hidden)

        # Reshape back to (num_tokens, top_k, hidden)
        expert_out = expert_out.view(
            num_tokens, self.num_experts_per_tok, self.hidden_size
        )

        # Weighted combination: sum over top_k dimension
        # topk_weight: (num_tokens, top_k) -> (num_tokens, top_k, 1)
        y = (expert_out * topk_weight.unsqueeze(-1)).sum(dim=1)  # (num_tokens, hidden)

        y = y.view(*orig_shape)

        # Add shared expert output
        if self.shared_gate_proj is not None:
            shared_out = self.shared_down_proj(
                F.silu(self.shared_gate_proj(identity)) * self.shared_up_proj(identity)
            )
            y = y + shared_out

        return y


# DeepSeek-V3 style configuration (scaled down for single H100)
# Full DeepSeek has 256 experts, we use 64 for manageable memory
batch_size = 4
seq_len = 2048
hidden_size = 2048
intermediate_size = 1408  # ~0.7x hidden for SwiGLU-style
n_routed_experts = 64
num_experts_per_tok = 8
n_group = 8  # 64 experts / 8 groups = 8 experts per group
topk_group = 4  # Select 4 groups out of 8
n_shared_experts = 2
routed_scaling_factor = 2.5


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [
        hidden_size,
        intermediate_size,
        n_routed_experts,
        num_experts_per_tok,
        n_group,
        topk_group,
        n_shared_experts,
        routed_scaling_factor,
    ]
