from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pycolmap
from torch.utils.data import Dataset


@dataclass
class Camera:
    id: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Image:
    path: Path
    camera_index: int
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector


class ColmapDataset(Dataset[Tuple[Image, Camera]]):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.sparse_dir = self.root_dir / "sparse" / "0"
        self.cameras: List[Camera] = []
        self.images: List[Image] = []
        self._load_data()
        # HACK: Search for most downscaled version...
        for scale in range(16):
            dir = self.root_dir / f"images_{scale}"
            if dir.exists():
                self.images_dir = dir

    def _load_data(self) -> None:
        """Load camera parameters and image paths from COLMAP format using pycolmap."""
        # Load reconstruction using pycolmap
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))

        # Convert cameras
        for camera_id, camera in reconstruction.cameras.items():
            if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                fx, cx, cy = camera.params
                fy = fx  # Assuming square pixels
            elif camera.model == pycolmap.CameraModelId.PINHOLE:
                fx, fy, cx, cy = camera.params
            else:
                raise ValueError(f"Unsupported camera model: {camera.model}")
            self.cameras.append(
                Camera(
                    id=camera_id,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
            )

        # Convert images
        for _, image in reconstruction.images.items():
            transform = image.cam_from_world
            self.images.append(
                Image(
                    path=self.images_dir / image.name,
                    camera_index=image.camera_id - 1,
                    R=np.array(transform.rotation.matrix(), dtype=np.float32),
                    t=np.array(transform.translation, dtype=np.float32),
                )
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Image, Camera]:
        image = self.images[idx]
        return image, self.cameras[image.camera_index]
