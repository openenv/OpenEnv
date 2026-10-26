"""
Block Matching Motion Estimation

Finds motion vectors between two video frames using block matching.
Core operation in video compression (H.264/H.265) and frame interpolation.

For each block in the current frame, searches for the best matching block
in a reference frame within a search range.

Optimization opportunities:
- Hierarchical search (coarse to fine)
- Early termination when good match found
- Shared memory for reference blocks
- SIMD SAD (Sum of Absolute Differences) computation
- Diamond or hexagonal search patterns
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Full-search block matching motion estimation.
    """

    def __init__(self, block_size: int = 16, search_range: int = 16):
        super(Model, self).__init__()
        self.block_size = block_size
        self.search_range = search_range

    def forward(
        self, current_frame: torch.Tensor, reference_frame: torch.Tensor
    ) -> tuple:
        """
        Estimate motion vectors between frames.

        Args:
            current_frame: (H, W) current frame
            reference_frame: (H, W) reference frame

        Returns:
            motion_x: (H//block_size, W//block_size) horizontal motion vectors
            motion_y: (H//block_size, W//block_size) vertical motion vectors
            sad: (H//block_size, W//block_size) minimum SAD for each block
        """
        H, W = current_frame.shape
        bs = self.block_size
        sr = self.search_range

        # Number of blocks
        num_blocks_y = H // bs
        num_blocks_x = W // bs

        # Output motion vectors
        motion_x = torch.zeros(num_blocks_y, num_blocks_x, device=current_frame.device)
        motion_y = torch.zeros(num_blocks_y, num_blocks_x, device=current_frame.device)
        min_sad = torch.full(
            (num_blocks_y, num_blocks_x), float("inf"), device=current_frame.device
        )

        # Pad reference frame for search
        ref_padded = torch.nn.functional.pad(
            reference_frame, (sr, sr, sr, sr), mode="constant", value=0
        )

        # For each block
        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                # Current block position
                cy = by * bs
                cx = bx * bs

                # Extract current block
                current_block = current_frame[cy : cy + bs, cx : cx + bs]

                # Search window in reference (accounting for padding)
                best_sad = float("inf")
                best_dx, best_dy = 0, 0

                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        # Reference block position (in padded coordinates)
                        ry = cy + sr + dy
                        rx = cx + sr + dx

                        # Extract reference block
                        ref_block = ref_padded[ry : ry + bs, rx : rx + bs]

                        # Compute SAD
                        sad = (current_block - ref_block).abs().sum()

                        if sad < best_sad:
                            best_sad = sad
                            best_dx, best_dy = dx, dy

                motion_x[by, bx] = best_dx
                motion_y[by, bx] = best_dy
                min_sad[by, bx] = best_sad

        return motion_x, motion_y, min_sad


# Problem configuration - HD frame
frame_height = 720
frame_width = 1280


def get_inputs():
    # Two consecutive frames
    current_frame = torch.rand(frame_height, frame_width)
    reference_frame = torch.rand(frame_height, frame_width)
    return [current_frame, reference_frame]


def get_init_inputs():
    return [16, 16]  # block_size, search_range
