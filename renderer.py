from typing import Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CameraParams:
    R: torch.Tensor  # 3x3 rotation matrix
    t: torch.Tensor  # 3x1 translation vector
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class GaussianRenderer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        means: torch.Tensor,
        covariance_matrices: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        camera: CameraParams,
    ) -> torch.Tensor:
        """Render the 3D Gaussians from the given camera viewpoint."""
        # Transform means to camera space
        means_cam = torch.matmul(camera.R, means.T).T + camera.t

        # Project means to image plane
        means_2d = self._project_points(means_cam, camera)

        # Transform covariance matrices to camera space
        cov_cam = torch.matmul(torch.matmul(camera.R, covariance_matrices), camera.R.T)

        # Project covariance matrices to 2D
        cov_2d = self._project_covariance(cov_cam, means_cam, camera)

        # Sort Gaussians by depth for back-to-front rendering
        depths = means_cam[:, 2]
        sorted_indices = torch.argsort(depths, descending=True)

        # Initialize output image and accumulated alpha
        image = torch.zeros((camera.height, camera.width, 3), device=means.device)
        accum_alpha = torch.zeros((camera.height, camera.width, 1), device=means.device)

        # Render each Gaussian
        for idx in sorted_indices:
            mean_2d = means_2d[idx]
            cov_2d_gaussian = cov_2d[idx]
            color = colors[idx]
            opacity = opacities[idx]

            # Skip if behind camera
            if depths[idx] <= 0:
                continue

            # Compute bounding box
            x_min, y_min, x_max, y_max = self._compute_bbox(
                mean_2d, cov_2d_gaussian, camera
            )
            if x_min >= x_max or y_min >= y_max:
                continue

            # Create pixel coordinates for bounding box region
            y, x = torch.meshgrid(
                torch.arange(y_min, y_max, device=means.device),
                torch.arange(x_min, x_max, device=means.device),
                indexing="ij",
            )
            coords = torch.stack([x, y], dim=-1)

            # Compute Gaussian weights
            weights = self._compute_gaussian_weights(coords, mean_2d, cov_2d_gaussian)

            # Compute alpha for this Gaussian
            alpha = weights.unsqueeze(-1) * opacity

            # Get current region (make a copy to avoid in-place operations)
            region_slice = (slice(y_min, y_max), slice(x_min, x_max))
            current_region = image[region_slice].clone()
            current_alpha = accum_alpha[region_slice].clone()

            # Compute new color using alpha compositing
            new_color = color.unsqueeze(0).unsqueeze(0) * alpha + current_region * (
                1 - alpha
            )
            new_alpha = alpha + current_alpha * (1 - alpha)

            # Update image and alpha directly
            image[region_slice] = new_color
            accum_alpha[region_slice] = new_alpha

        return image

    def _project_points(
        self, points: torch.Tensor, camera: CameraParams
    ) -> torch.Tensor:
        """Project 3D points to 2D image plane."""
        # Apply perspective projection
        z = points[:, 2]
        x = points[:, 0] / z
        y = points[:, 1] / z

        # Apply camera intrinsics
        u = x * camera.fx + camera.cx
        v = y * camera.fy + camera.cy

        return torch.stack([u, v], dim=1)

    def _project_covariance(
        self, cov_3d: torch.Tensor, points: torch.Tensor, camera: CameraParams
    ) -> torch.Tensor:
        """Project 3D covariance matrices to 2D."""
        # Get camera center
        camera_center = -torch.matmul(camera.R.T, camera.t)

        # Compute Jacobian of projection
        z = points[:, 2]
        x = points[:, 0]
        y = points[:, 1]

        # Create zero tensors with correct shape and device
        zeros = torch.zeros_like(z)

        J = torch.stack(
            [
                torch.stack([camera.fx / z, zeros, -camera.fx * x / (z * z)], dim=1),
                torch.stack([zeros, camera.fy / z, -camera.fy * y / (z * z)], dim=1),
            ],
            dim=1,
        )

        # Project covariance: J Σ J^T
        cov_2d = torch.matmul(torch.matmul(J, cov_3d), J.transpose(-2, -1))

        return cov_2d

    def _compute_bbox(
        self, mean_2d: torch.Tensor, cov_2d: torch.Tensor, camera: CameraParams
    ) -> Tuple[int, int, int, int]:
        """Compute bounding box for a 2D Gaussian."""
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(cov_2d)

        # Compute bounding box size (3 standard deviations)
        size = 3 * torch.sqrt(
            eigenvals.abs()
        )  # Use abs to handle numerical instability

        # Compute corners
        corners = torch.stack(
            [
                mean_2d + size[0].unsqueeze(0) * eigenvecs[:, 0],
                mean_2d - size[0].unsqueeze(0) * eigenvecs[:, 0],
                mean_2d + size[1].unsqueeze(0) * eigenvecs[:, 1],
                mean_2d - size[1].unsqueeze(0) * eigenvecs[:, 1],
            ]
        )

        # Get bounding box
        x_min, y_min = corners.min(dim=0)[0]
        x_max, y_max = corners.max(dim=0)[0]

        # Convert to integers for indexing
        x_min_int = max(0, int(torch.floor(x_min).item()))
        y_min_int = max(0, int(torch.floor(y_min).item()))
        x_max_int = min(camera.width, int(torch.ceil(x_max).item()))
        y_max_int = min(camera.height, int(torch.ceil(y_max).item()))

        return x_min_int, y_min_int, x_max_int, y_max_int

    def _compute_gaussian_weights(
        self, coords: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gaussian weights for given coordinates."""
        # Compute inverse covariance
        cov_inv = torch.linalg.inv(cov)

        # Compute Mahalanobis distance
        diff = coords - mean
        dist = torch.sum(torch.matmul(diff, cov_inv) * diff, dim=-1)

        # Compute Gaussian weights
        weights = torch.exp(-0.5 * dist)

        return weights
