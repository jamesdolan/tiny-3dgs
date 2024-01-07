from functools import lru_cache
from pathlib import Path
from plyfile import PlyData, PlyElement  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import ColmapDataset, Camera, Image
from gaussian_model import GaussianModel
from renderer import GaussianRenderer, CameraParams


@lru_cache(maxsize=256)
def load_image(device: torch.device, path: Path) -> torch.Tensor:
    image = (
        torch.from_numpy(np.array(PIL.Image.open(path).convert("RGB"))).float() / 255.0
    )
    image = image.permute(2, 0, 1)  # HWC to CHW
    return image.to(device)


def collate_fn(batch: List[Tuple[Image, Camera]]) -> Tuple[List[Image], List[Camera]]:
    """Custom collate function to handle Image and Camera objects."""
    images, cameras = zip(*batch)
    return list(images), list(cameras)


def convert_camera_to_params(
    camera: Camera, image: Image, img_tensor: torch.Tensor, device: torch.device
) -> CameraParams:
    """Convert Camera and Image objects to CameraParams."""
    # Load image to get dimensions
    height, width = img_tensor.shape[1:]  # CHW format

    return CameraParams(
        R=torch.from_numpy(image.R).float().to(device),
        t=torch.from_numpy(image.t).float().to(device),
        fx=float(camera.fx),
        fy=float(camera.fy),
        cx=float(camera.cx),
        cy=float(camera.cy),
        width=width,
        height=height,
    )


def save_to_ply(model: GaussianModel, output_path: str) -> None:
    """Save the trained Gaussian model to a PLY file."""
    params = model.get_params()

    # Convert parameters to numpy arrays
    means = params.means.detach().cpu().numpy()
    scales = params.scales.detach().cpu().numpy()
    rotations = params.rotations.detach().cpu().numpy()
    colors = params.colors.detach().cpu().numpy()
    opacities = params.opacities.detach().cpu().numpy()

    # Create vertex element
    vertex = np.zeros(
        len(means),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
        ],
    )

    vertex["x"] = means[:, 0]
    vertex["y"] = means[:, 1]
    vertex["z"] = means[:, 2]
    vertex["scale_0"] = scales[:, 0]
    vertex["scale_1"] = scales[:, 1]
    vertex["scale_2"] = scales[:, 2]
    vertex["rot_0"] = rotations[:, 0]  # w component
    vertex["rot_1"] = rotations[:, 1]  # x component
    vertex["rot_2"] = rotations[:, 2]  # y component
    vertex["rot_3"] = rotations[:, 3]  # z component
    vertex["f_dc_0"] = colors[:, 0]
    vertex["f_dc_1"] = colors[:, 1]
    vertex["f_dc_2"] = colors[:, 2]
    vertex["opacity"] = opacities[:, 0]

    # Create PLY element
    el = PlyElement.describe(vertex, "vertex")

    # Save to file
    PlyData([el]).write(output_path)


def train(
    dataset_path: str,
    num_gaussians: int = 100000,
    num_epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 0.001,
    output_dir: str = "output",
) -> None:
    """Train the 3D Gaussian Splatting model."""
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    # Load dataset
    dataset = ColmapDataset(dataset_path)
    dataloader: DataLoader[Tuple[Image, Camera]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialize model and renderer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaussianModel(num_gaussians).to(device)
    renderer = GaussianRenderer().to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, cameras in progress_bar:
            # Move images to device
            loaded_images: List[torch.Tensor] = [
                load_image(device, img.path) for img in images
            ]
            loaded_images_tensor = torch.stack(loaded_images)

            # Convert cameras to params
            camera_params = [
                convert_camera_to_params(cam, img, img_tensor, device)
                for cam, img, img_tensor in zip(cameras, images, loaded_images_tensor)
            ]

            # Get model parameters
            means = model.get_means()
            covariance_matrices = model.get_covariance_matrices()
            colors = model.get_colors()
            opacities = model.get_opacities()

            # Render image
            rendered = renderer(
                means,
                covariance_matrices,
                colors,
                opacities,
                camera_params[0],  # For now, just use first camera in batch
            )

            # Compute loss
            target = loaded_images_tensor[0].permute(1, 2, 0)  # Convert from CHW to HWC
            loss = criterion(rendered, target)  # Both should be (H, W, C)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Save intermediate results
        save_rate = 1
        if (epoch + 1) % save_rate == 0:
            # Save rendered image
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(loaded_images_tensor[0].cpu().permute(1, 2, 0))
            plt.title("Ground Truth")
            plt.subplot(1, 2, 2)
            plt.imshow(rendered.detach().cpu())
            plt.title("Rendered")
            plt.savefig(output_dir_path / f"epoch_latest.png")
            plt.close()

            # Save model
            save_to_ply(model, str(output_dir_path / f"model_epoch_latest.ply"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to COLMAP dataset"
    )
    parser.add_argument(
        "--num_gaussians", type=int, default=1000, help="Number of Gaussians"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )

    args = parser.parse_args()
    train(**vars(args))
