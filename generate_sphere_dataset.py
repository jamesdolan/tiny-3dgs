#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Final, List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pycolmap
from PIL import Image
from tqdm import tqdm

NDArrayFloat = npt.NDArray[np.float64]


def ray_sphere_intersection(
    ray_origin: NDArrayFloat,
    ray_direction: NDArrayFloat,
    sphere_center: NDArrayFloat,
    sphere_radius: float,
) -> tuple[float | None, NDArrayFloat | None]:
    """Compute ray-sphere intersection.

    Returns:
        Tuple of (t, normal) where t is the distance along the ray to intersection,
        or (None, None) if no intersection.
    """
    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None, None

    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        return None, None

    intersection = ray_origin + t * ray_direction
    normal = (intersection - sphere_center) / sphere_radius

    return t, normal


def create_camera_rays(width: int, height: int, fov: float) -> NDArrayFloat:
    """Create ray directions for each pixel in the camera."""
    # Convert FOV to radians
    fov_rad = math.radians(fov)

    # Calculate focal length
    focal_length = width / (2 * math.tan(fov_rad / 2))

    # Create pixel coordinates
    y, x = np.mgrid[-height // 2 : height // 2, -width // 2 : width // 2]

    # Create ray directions (normalized)
    rays = np.stack([x / focal_length, y / focal_length, np.ones_like(x)], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    return cast(NDArrayFloat, rays)


def render_sphere(
    width: int,
    height: int,
    fov: float,
    camera_position: NDArrayFloat,
    camera_rotation: NDArrayFloat,
    sphere_center: NDArrayFloat = np.array([0.0, 0.0, 0.0], dtype=np.float64),
    sphere_radius: float = 1.0,
) -> NDArrayFloat:
    """Render a sphere from a given camera position."""
    # Create ray directions for each pixel
    rays = create_camera_rays(width, height, fov)

    # Transform rays to world space (camera_rotation is world_from_cam)
    rays = rays @ camera_rotation

    # Initialize image
    image = np.zeros((height, width, 3), dtype=np.float64)

    # For each pixel
    for y in range(height):
        for x in range(width):
            # Get ray direction
            ray_dir = rays[y, x]

            # Compute intersection
            t, normal = ray_sphere_intersection(
                camera_position, ray_dir, sphere_center, sphere_radius
            )

            if t is not None and normal is not None:
                # Color based on normal
                color = (normal + 1.0) / 2.0  # Map from [-1,1] to [0,1]
                image[y, x] = color

    return image


def create_camera_poses(
    num_cameras: int, radius: float = 3.0, height: float = 0.0
) -> List[Tuple[NDArrayFloat, NDArrayFloat]]:
    """Generate camera positions and orientations in a circle around the sphere."""
    positions: List[Tuple[NDArrayFloat, NDArrayFloat]] = []

    for i in range(num_cameras):
        angle = 2 * math.pi * i / num_cameras
        x = radius * math.cos(angle)
        y = height
        z = radius * math.sin(angle)
        position = np.array([x, y, z], dtype=np.float64)

        # Camera looking at center
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Compute camera orientation
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Create rotation matrix (right, up, forward)
        rotation = np.stack([right, up, forward], axis=1)
        positions.append((position, rotation.T))

    return positions


def generate_colmap_dataset(
    output_dir: Path,
    num_cameras: int,
    width: int = 800,
    height: int = 600,
    fov: float = 60.0,
) -> None:
    """Generate a COLMAP-compatible dataset of a sphere."""
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Generate camera positions
    camera_poses = create_camera_poses(num_cameras)

    # Prepare COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add camera
    camera = pycolmap.Camera.create(
        camera_id=1,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=width / (2 * math.tan(math.radians(fov) / 2)),
        width=width,
        height=height,
    )
    reconstruction.add_camera(camera)

    # Render and save images
    for i, (position, rotation) in enumerate(
        tqdm(camera_poses, desc="Rendering images")
    ):
        # Render image
        rendered_image = render_sphere(width, height, fov, position, rotation)

        # Save image
        image_path = images_dir / f"image_{i:04d}.jpg"
        Image.fromarray((rendered_image * 255).astype(np.uint8)).save(
            image_path, quality=95
        )

        # Create image object
        colmap_image = pycolmap.Image(image_id=i + 1, name=image_path.name, camera_id=1)

        # Set the pose using the transform
        transform = pycolmap.Rigid3d()
        transform.rotation = pycolmap.Rotation3d(rotation)
        transform.translation = rotation @ -position
        colmap_image.cam_from_world = transform

        # Add image to reconstruction
        reconstruction.add_image(colmap_image)

    # Save reconstruction in text format
    reconstruction.write_text(str(sparse_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic sphere dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sphere"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--num-cameras", type=int, default=16, help="Number of cameras to generate"
    )
    parser.add_argument("--width", type=int, default=128, help="Image width")
    parser.add_argument("--height", type=int, default=128, help="Image height")
    parser.add_argument(
        "--fov", type=float, default=60.0, help="Field of view in degrees"
    )

    args = parser.parse_args()

    generate_colmap_dataset(
        args.output, args.num_cameras, args.width, args.height, args.fov
    )

    print(f"Dataset generated in {args.output}")


if __name__ == "__main__":
    main()
