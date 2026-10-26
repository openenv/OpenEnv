"""
Ray Tracing - Sphere Intersection

Traces rays against a scene of spheres and computes intersections.
This is the core operation in ray tracing renderers.

Challenge: Divergent control flow as rays hit different objects at different depths.

Optimization opportunities:
- Ray packet tracing (process multiple rays together)
- Persistent threads with ray queues
- Warp-coherent intersection testing
- SIMD sphere testing (test 4 spheres per iteration)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Ray-sphere intersection testing.

    For each ray, finds the closest sphere intersection.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        sphere_centers: torch.Tensor,
        sphere_radii: torch.Tensor,
    ) -> tuple:
        """
        Find closest ray-sphere intersection for each ray.

        Args:
            ray_origins: (N, 3) ray origins
            ray_directions: (N, 3) ray directions (normalized)
            sphere_centers: (M, 3) sphere centers
            sphere_radii: (M,) sphere radii

        Returns:
            t_hit: (N,) distance to closest hit (inf if no hit)
            sphere_idx: (N,) index of hit sphere (-1 if no hit)
            hit_points: (N, 3) intersection points
            hit_normals: (N, 3) surface normals at hit points
        """
        N = ray_origins.shape[0]
        M = sphere_centers.shape[0]

        # Initialize outputs
        t_hit = torch.full((N,), float("inf"), device=ray_origins.device)
        sphere_idx = torch.full((N,), -1, dtype=torch.long, device=ray_origins.device)

        # Brute force: test each ray against each sphere
        for i in range(N):
            origin = ray_origins[i]
            direction = ray_directions[i]

            for j in range(M):
                center = sphere_centers[j]
                radius = sphere_radii[j]

                # Ray-sphere intersection using quadratic formula
                # Ray: P = O + t*D
                # Sphere: |P - C|^2 = r^2
                # Substituting: |O + t*D - C|^2 = r^2
                # Let L = O - C
                # |L + t*D|^2 = r^2
                # t^2*(D.D) + 2t*(D.L) + (L.L - r^2) = 0

                L = origin - center
                a = torch.dot(direction, direction)
                b = 2.0 * torch.dot(direction, L)
                c = torch.dot(L, L) - radius * radius

                discriminant = b * b - 4 * a * c

                if discriminant >= 0:
                    sqrt_disc = torch.sqrt(discriminant)
                    t1 = (-b - sqrt_disc) / (2 * a)
                    t2 = (-b + sqrt_disc) / (2 * a)

                    # Take closest positive t
                    t = t1 if t1 > 0 else t2

                    if t > 0 and t < t_hit[i]:
                        t_hit[i] = t
                        sphere_idx[i] = j

        # Compute hit points and normals
        hit_points = ray_origins + t_hit.unsqueeze(1) * ray_directions
        hit_normals = torch.zeros_like(hit_points)

        for i in range(N):
            if sphere_idx[i] >= 0:
                center = sphere_centers[sphere_idx[i]]
                hit_normals[i] = hit_points[i] - center
                hit_normals[i] = hit_normals[i] / hit_normals[i].norm()

        return t_hit, sphere_idx, hit_points, hit_normals


# Problem configuration
num_rays = 65536  # 256x256 image
num_spheres = 256


def get_inputs():
    # Camera rays (simple pinhole camera looking at origin)
    # Create a grid of rays
    W, H = 256, 256
    u = torch.linspace(-1, 1, W)
    v = torch.linspace(-1, 1, H)
    U, V = torch.meshgrid(u, v, indexing="ij")

    # Ray origins at z=5
    ray_origins = torch.zeros(num_rays, 3)
    ray_origins[:, 2] = 5.0

    # Ray directions towards image plane at z=0
    ray_directions = torch.zeros(num_rays, 3)
    ray_directions[:, 0] = U.flatten()
    ray_directions[:, 1] = V.flatten()
    ray_directions[:, 2] = -1.0
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

    # Random spheres in the scene
    sphere_centers = torch.randn(num_spheres, 3) * 2
    sphere_radii = torch.rand(num_spheres) * 0.5 + 0.1

    return [ray_origins, ray_directions, sphere_centers, sphere_radii]


def get_init_inputs():
    return []
