"""FFHQ-Wrinkle images, texture maps, masks, and split IDs."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLIT_DIR = Path(__file__).parent / "splits" / "ffhq"


def read_ids(path: str | Path) -> list[str]:
    ids = [Path(line.strip()).stem for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split must contain unique, nonempty IDs: {path}")
    return ids


def index_pngs(directory: str | Path) -> dict[str, Path]:
    """Index PNGs recursively by filename stem."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = {}
    for path in sorted(directory.rglob("*.png")):
        if path.stem in files:
            raise ValueError(f"Duplicate image ID {path.stem}: {files[path.stem]} and {path}")
        files[path.stem] = path
    return files


def load_pair(image_path: Path, texture_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    with Image.open(texture_path) as image:
        texture = np.array(image.convert("L"))
    if rgb.shape[:2] != texture.shape:
        raise ValueError(f"Image/texture size mismatch for {image_path.stem}")
    return rgb, texture


def input_tensor(rgb: np.ndarray, texture: np.ndarray) -> torch.Tensor:
    """Stack RGB and texture channels and map uint8 values to [-1, 1]."""
    combined = np.concatenate((rgb, texture[..., None]), axis=-1)
    return torch.from_numpy(combined.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1)


def _augmentations(image_size):
    import albumentations as A
    import cv2

    spatial = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                image_size, image_size, scale=(0.67, 1.33), ratio=(0.75, 4 / 3), p=1
            ),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                rotate=(-30, 30),
                shear={"x": (-8, 8), "y": (-8, 8)},
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                approximate=True,
                same_dxdy=True,
                p=0.5,
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.OpticalDistortion(
                distort_limit=0.1,
                shift_limit=0.1,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
        ]
    )
    color = A.Compose([A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5)])
    return spatial, color


class FFHQWrinkle(Dataset):
    """RGB and texture inputs paired with binary wrinkle masks.

    Only the training split is augmented.
    """

    def __init__(self, root, split="train", split_dir=SPLIT_DIR, image_size=1024):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if image_size < 64 or image_size % 32:
            raise ValueError("image_size must be a multiple of 32 and at least 64")
        self.ids = read_ids(Path(split_dir) / f"{split}.txt")
        self.image_size = image_size
        self.training = split == "train"
        root = Path(root)
        self.images = index_pngs(root / "masked_face_images")
        self.textures = index_pngs(root / "weak_wrinkle_masks")
        self.masks = index_pngs(root / "manual_wrinkle_masks")
        for label, files in (
            ("images", self.images),
            ("textures", self.textures),
            ("masks", self.masks),
        ):
            missing = [case for case in self.ids if case not in files]
            if missing:
                raise FileNotFoundError(
                    f"Missing {label} for {len(missing)} IDs: {', '.join(missing[:10])}"
                )
        self.spatial, self.color = _augmentations(image_size) if self.training else (None, None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        case = self.ids[index]
        rgb, texture = load_pair(self.images[case], self.textures[case])
        with Image.open(self.masks[case]) as image:
            mask = (np.array(image.convert("L")) > 0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape or mask.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected aligned {self.image_size}x{self.image_size} inputs for {case}, got {rgb.shape[:2]}, {mask.shape}"
            )
        if self.training:
            transformed = self.spatial(image=rgb, masks=[mask, texture])
            rgb = self.color(image=transformed["image"])["image"]
            mask, texture = transformed["masks"]
        return {
            "id": case,
            "image": input_tensor(rgb, texture),
            "mask": torch.from_numpy(mask.copy()).long(),
        }


def validate_splits(split_dir=SPLIT_DIR):
    splits = {
        name: set(read_ids(Path(split_dir) / f"{name}.txt")) for name in ("train", "val", "test")
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if splits[first] & splits[second]:
            raise ValueError(f"Overlapping IDs in {first}/{second} splits")
    return {name: len(ids) for name, ids in splits.items()}
