"""
Bounding Volume Hierarchy (BVH) Traversal for Ray-Box Intersection

Tests rays against axis-aligned bounding boxes organized in a BVH tree.
This is a key operation in ray tracing and collision detection.

Challenge: Divergent control flow as different rays traverse different paths.

Optimization opportunities:
- Stackless traversal algorithms
- Ray packet tracing (SIMD over rays)
- Warp-coherent traversal
- Persistent threads with work stealing
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    BVH traversal for ray-AABB intersection testing.

    Each ray tests against a binary BVH tree of AABBs.
    Returns the closest intersection distance (or inf if no hit).
    """

    def __init__(self, num_nodes: int = 1023):
        super(Model, self).__init__()
        self.num_nodes = num_nodes  # 2^10 - 1 for 10 levels

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        bvh_min: torch.Tensor,
        bvh_max: torch.Tensor,
        bvh_left: torch.Tensor,
        bvh_right: torch.Tensor,
        bvh_is_leaf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Traverse BVH for each ray and return closest intersection.

        Args:
            ray_origins: (N, 3) ray origins
            ray_directions: (N, 3) ray directions (normalized)
            bvh_min: (num_nodes, 3) AABB minimum corners
            bvh_max: (num_nodes, 3) AABB maximum corners
            bvh_left: (num_nodes,) left child indices (-1 for none)
            bvh_right: (num_nodes,) right child indices (-1 for none)
            bvh_is_leaf: (num_nodes,) whether node is a leaf

        Returns:
            t_hit: (N,) intersection distances (inf if no hit)
        """
        N = ray_origins.shape[0]
        t_hit = torch.full((N,), float("inf"), device=ray_origins.device)

        # Process each ray
        for ray_idx in range(N):
            origin = ray_origins[ray_idx]
            direction = ray_directions[ray_idx]
            inv_dir = 1.0 / (direction + 1e-10)  # Avoid division by zero

            # Stack-based traversal
            stack = [0]  # Start at root
            closest_t = float("inf")

            while stack:
                node_idx = stack.pop()
                if node_idx < 0 or node_idx >= self.num_nodes:
                    continue

                # Ray-AABB intersection test
                t_min_box = (bvh_min[node_idx] - origin) * inv_dir
                t_max_box = (bvh_max[node_idx] - origin) * inv_dir

                # Handle negative directions
                t1 = torch.minimum(t_min_box, t_max_box)
                t2 = torch.maximum(t_min_box, t_max_box)

                t_near = t1.max().item()
                t_far = t2.min().item()

                # Check for intersection
                if t_near <= t_far and t_far >= 0 and t_near < closest_t:
                    if bvh_is_leaf[node_idx]:
                        # Leaf node - record hit
                        hit_t = max(0, t_near)
                        closest_t = min(closest_t, hit_t)
                    else:
                        # Internal node - push children
                        left = bvh_left[node_idx].item()
                        right = bvh_right[node_idx].item()
                        if left >= 0:
                            stack.append(left)
                        if right >= 0:
                            stack.append(right)

            t_hit[ray_idx] = closest_t

        return t_hit


# Problem configuration
num_rays = 65536
num_bvh_nodes = 1023  # Binary tree with 10 levels


def get_inputs():
    # Random rays
    ray_origins = torch.randn(num_rays, 3)
    ray_directions = torch.randn(num_rays, 3)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

    # Build random BVH (not a valid spatial hierarchy, just for testing)
    bvh_min = torch.randn(num_bvh_nodes, 3) - 1
    bvh_max = bvh_min + torch.rand(num_bvh_nodes, 3) * 2

    # Build tree structure
    bvh_left = torch.zeros(num_bvh_nodes, dtype=torch.long)
    bvh_right = torch.zeros(num_bvh_nodes, dtype=torch.long)
    bvh_is_leaf = torch.zeros(num_bvh_nodes, dtype=torch.bool)

    for i in range(num_bvh_nodes):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < num_bvh_nodes:
            bvh_left[i] = left_child
            bvh_right[i] = right_child
        else:
            bvh_left[i] = -1
            bvh_right[i] = -1
            bvh_is_leaf[i] = True

    return [
        ray_origins,
        ray_directions,
        bvh_min,
        bvh_max,
        bvh_left,
        bvh_right,
        bvh_is_leaf,
    ]


def get_init_inputs():
    return [num_bvh_nodes]
