import os
import json
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from PIL import Image


class MalariaDataset(Dataset):
    """
    Reusable Malaria detection dataset.

    Args:
        json_path (str): Path to annotations JSON.
        image_root (str): Directory containing images.
        transform (callable, optional): Image-only transform (e.g., ToTensor+Normalize).
            IMPORTANT: if you also resize in `transform`, DO NOT set `resize_to`, or
            you'll resize twice without re-scaling boxes.
        category_map (dict, optional): {category_name -> index}. If None, built from JSON (sorted).
        image_size (int): Used only for placeholder tensor if a file is missing.
        resize_to (int or (H,W), optional): If set, images are resized inside __getitem__
            and bounding boxes are scaled accordingly.
        return_path (bool): If True, returns (image, target, path). Otherwise (image, target).
        skip_empty (bool): If True, drops entries with no objects.

    Returns:
        image (Tensor CxHxW),
        target (dict): {'boxes': FloatTensor[N,4] in (x_min,y_min,x_max,y_max), 'labels': LongTensor[N]},
        [path (str)] if return_path=True
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        transform=None,
        category_map: Optional[Dict[str, int]] = None,
        image_size: int = 128,
        resize_to: Optional[Union[int, Tuple[int, int]]] = None,
        return_path: bool = False,
        skip_empty: bool = False,
    ):
        with open(json_path, "r") as f:
            entries = json.load(f)

        if skip_empty:
            entries = [e for e in entries if e.get("objects")]

        self.entries = entries
        self.image_root = image_root
        self.transform = transform
        self.image_size = int(image_size)
        self.return_path = return_path

        # Handle resize target
        if resize_to is not None and isinstance(resize_to, int):
            resize_to = (resize_to, resize_to)
        self.resize_to = resize_to  # (H, W) or None

        # Build / accept category map
        if category_map is None:
            cats = set()
            for item in self.entries:
                for obj in item.get("objects", []):
                    cats.add(obj["category"])
            self.category_map = {c: i for i, c in enumerate(sorted(cats))}
        else:
            self.category_map = dict(category_map)

        # Precompute a single label per entry (useful for weighted sampling etc.)
        self.labels = []
        for item in self.entries:
            objs = item.get("objects", [])
            if objs:
                self.labels.append(self.category_map[objs[0]["category"]])
            else:
                self.labels.append(-1)  # placeholder if keeping empties

    def __len__(self) -> int:
        return len(self.entries)

    def _scale_boxes(self, boxes: torch.Tensor, orig_size: Tuple[int, int], new_size: Tuple[int, int]) -> torch.Tensor:
        """Scale boxes from (orig_h,orig_w) -> (new_h,new_w) inplace-safe."""
        if boxes.numel() == 0:
            return boxes
        orig_h, orig_w = orig_size
        new_h, new_w = new_size
        sx = new_w / max(orig_w, 1e-12)
        sy = new_h / max(orig_h, 1e-12)
        boxes = boxes.clone()
        # boxes: [x_min, y_min, x_max, y_max]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy
        # clip to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h - 1)
        return boxes

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        # image path (robust to path containing directories)
        image_name = os.path.basename(entry["image"]["pathname"])
        image_full_path = os.path.join(self.image_root, image_name)

        # load image or placeholder
        try:
            img = Image.open(image_full_path).convert("RGB")
            orig_w, orig_h = img.size
        except FileNotFoundError:
            # Placeholder if missing
            img_tensor = torch.zeros((3, self.image_size, self.image_size))
            target = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
            if self.return_path:
                return img_tensor, target, image_full_path
            return img_tensor, target

        # parse boxes/labels
        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in entry.get("objects", []):
            bb = obj["bounding_box"]
            # JSON appears to store as columns (c) and rows (r): (x,y)
            x_min, y_min = bb["minimum"]["c"], bb["minimum"]["r"]
            x_max, y_max = bb["maximum"]["c"], bb["maximum"]["r"]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.category_map[obj["category"]])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long)

        # Optional resize here (and scale boxes)
        if self.resize_to is not None:
            new_h, new_w = int(self.resize_to[0]), int(self.resize_to[1])
            if (orig_h, orig_w) != (new_h, new_w):
                boxes = self._scale_boxes(boxes, (orig_h, orig_w), (new_h, new_w))
                img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Apply user transforms (image-only)
        if self.transform is not None:
            img = self.transform(img)

        target = {"boxes": boxes, "labels": labels}

        if self.return_path:
            return img, target, image_full_path
        return img, target


def detection_collate(batch):
    """
    Collate for variable-length targets:
    - images -> stacked tensor [B,C,H,W]
    - targets -> list of dicts (len B)
    - optionally paths if dataset.return_path=True
    """
    has_path = len(batch[0]) == 3
    images, targets, paths = [], [], []
    for item in batch:
        if has_path:
            img, tgt, p = item
            paths.append(p)
        else:
            img, tgt = item
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    if has_path:
        return images, targets, paths
    return images, targets
