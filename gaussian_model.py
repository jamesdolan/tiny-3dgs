from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GaussianParams:
    means: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3)
    rotations: torch.Tensor  # (N, 4) quaternions
    colors: torch.Tensor  # (N, 3)
    opacities: torch.Tensor  # (N, 1)


class GaussianModel(nn.Module):
    def __init__(self, num_gaussians: int):
        super().__init__()
        self.num_gaussians = num_gaussians

        # Initialize means uniformly in a sphere of radius 1.0
        # First generate random directions
        directions = torch.randn(num_gaussians, 3)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        # Then generate random radii between 0 and 1
        radii = torch.rand(num_gaussians, 1).pow(
            1 / 3
        )  # Cube root for uniform volume distribution
        self.means = nn.Parameter(directions * radii)

        # Initialize scales to reasonable size (about 0.1 of sphere radius)
        self.scales = nn.Parameter(
            torch.full((num_gaussians, 3), -2.3)
        )  # exp(-2.3) ≈ 0.1

        # Initialize rotations as identity quaternions with small noise
        self.rotations = nn.Parameter(
            torch.cat(
                [
                    torch.ones(num_gaussians, 1),  # w component
                    torch.zeros(num_gaussians, 3),  # x, y, z components
                ],
                dim=1,
            )
            + torch.randn(num_gaussians, 4) * 0.01
        )

        # Initialize colors to white with small noise
        self.colors = nn.Parameter(
            torch.ones(num_gaussians, 3) + torch.randn(num_gaussians, 3) * 0.1
        )

        # Initialize opacities to moderate values
        self.opacities = nn.Parameter(torch.zeros(num_gaussians, 1))  # sigmoid(0) = 0.5

    def get_params(self) -> GaussianParams:
        """Get all parameters as a GaussianParams object."""
        # Clamp scales to prevent explosion
        scales = torch.exp(self.scales.clamp(-10.0, 2.0))

        # Normalize quaternions
        rotations = self.rotations / (
            torch.norm(self.rotations, dim=1, keepdim=True) + 1e-8
        )

        # Ensure colors and opacities are in [0, 1]
        colors = torch.sigmoid(self.colors)
        opacities = torch.sigmoid(self.opacities)

        return GaussianParams(
            means=self.means,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
        )

    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices in a numerically stable way.

        Args:
            quaternions: (N, 4) tensor of quaternions (w, x, y, z)

        Returns:
            (N, 3, 3) tensor of rotation matrices
        """
        # Normalize quaternions
        norm = torch.norm(quaternions, dim=1, keepdim=True)
        quaternions = quaternions / norm

        # Extract components
        w, x, y, z = quaternions.unbind(dim=1)

        # Compute rotation matrix elements
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)

        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)

        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        # Stack into matrix
        rotation_matrix = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=1),
                torch.stack([r10, r11, r12], dim=1),
                torch.stack([r20, r21, r22], dim=1),
            ],
            dim=1,
        )

        return rotation_matrix

    def get_covariance_matrices(self) -> torch.Tensor:
        """Compute 3x3 covariance matrices for each Gaussian."""
        # Get normalized parameters
        params = self.get_params()

        # Convert quaternions to rotation matrices
        rotation_matrices = self._quaternion_to_rotation_matrix(params.rotations)

        # Create scale matrices
        scale_matrices = torch.diag_embed(params.scales)

        # Compute covariance matrices: R S S^T R^T
        # Use separate operations to maintain numerical stability
        RS = torch.matmul(rotation_matrices, scale_matrices)
        RSS = torch.matmul(RS, scale_matrices)
        RSSR = torch.matmul(RSS, rotation_matrices.transpose(-2, -1))

        return RSSR

    def get_means(self) -> torch.Tensor:
        """Get the means of the Gaussians."""
        return self.means

    def get_colors(self) -> torch.Tensor:
        """Get the colors of the Gaussians."""
        return torch.sigmoid(self.colors)

    def get_opacities(self) -> torch.Tensor:
        """Get the opacities of the Gaussians."""
        return torch.sigmoid(self.opacities)
