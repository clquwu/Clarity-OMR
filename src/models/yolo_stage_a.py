#!/usr/bin/env python3
"""Stage A page analysis with YOLOv8m for OMR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class RegionClass(IntEnum):
    STAFF = 0
    SYSTEM_BRACKET = 1
    BARLINE_SYSTEM = 2
    TITLE_REGION = 3
    PAGE_NUMBER = 4


DETECTION_CLASS_NAMES = [
    "staff",
]


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def x_center(self) -> float:
        return self.x_min + self.width / 2.0

    @property
    def y_center(self) -> float:
        return self.y_min + self.height / 2.0

    def clip(self, image_width: float, image_height: float) -> "BoundingBox":
        return BoundingBox(
            x_min=max(0.0, min(self.x_min, image_width)),
            y_min=max(0.0, min(self.y_min, image_height)),
            x_max=max(0.0, min(self.x_max, image_width)),
            y_max=max(0.0, min(self.y_max, image_height)),
        )

    def padded(self, vertical_ratio: float, horizontal_ratio: float) -> "BoundingBox":
        x_pad = self.width * horizontal_ratio
        y_pad = self.height * vertical_ratio
        return BoundingBox(
            x_min=self.x_min - x_pad,
            y_min=self.y_min - y_pad,
            x_max=self.x_max + x_pad,
            y_max=self.y_max + y_pad,
        )

    def overlaps_vertically(self, other: "BoundingBox", threshold: float) -> bool:
        overlap_top = max(self.y_min, other.y_min)
        overlap_bottom = min(self.y_max, other.y_max)
        overlap = max(0.0, overlap_bottom - overlap_top)
        normalizer = max(1e-6, min(self.height, other.height))
        return (overlap / normalizer) >= threshold

    def vertical_overlap_ratio(self, other: "BoundingBox") -> float:
        overlap_top = max(self.y_min, other.y_min)
        overlap_bottom = min(self.y_max, other.y_max)
        overlap = max(0.0, overlap_bottom - overlap_top)
        normalizer = max(1e-6, min(self.height, other.height))
        return overlap / normalizer

    def intersection_area(self, other: "BoundingBox") -> float:
        inter_x0 = max(self.x_min, other.x_min)
        inter_y0 = max(self.y_min, other.y_min)
        inter_x1 = min(self.x_max, other.x_max)
        inter_y1 = min(self.y_max, other.y_max)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        return inter_w * inter_h

    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = max(1e-9, self.area() + other.area() - inter)
        return inter / union


@dataclass(frozen=True)
class Detection:
    region_class: RegionClass
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class StaffCrop:
    source_image: str
    crop_path: str
    system_index: int
    staff_index: int
    bbox: BoundingBox


@dataclass
class YoloStageAConfig:
    merge_system_staves: bool = True
    weights_path: Optional[Path] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    system_overlap_threshold: float = 0.50
    vertical_padding_ratio: float = 0.30
    horizontal_padding_ratio: float = 0.03
    dedupe_iou_threshold: float = 0.85
    dedupe_vertical_overlap_threshold: float = 0.75
    dedupe_center_distance_ratio: float = 0.40
    enforce_full_width_crops: bool = False
    full_width_left_page_edge: bool = True
    full_width_right_page_edge: bool = False
    min_vertical_padding_px: float = 12.0
    min_right_padding_px: float = 48.0
    seed: int = 1337


def _deterministic_split(sample_id: str, seed: int, train: float, val: float) -> str:
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train:
        return "train"
    if value < train + val:
        return "val"
    return "test"


class YoloStageA:
    def __init__(self, config: Optional[YoloStageAConfig] = None) -> None:
        self.config = config or YoloStageAConfig()
        self._model = None

    def load_model(self) -> None:
        if self.config.weights_path is None:
            raise ValueError("weights_path is required to load YOLO model.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for Stage A. Install with: pip install ultralytics") from exc
        self._model = YOLO(str(self.config.weights_path))

    def detect_regions(self, image_path: Path) -> List[Detection]:
        if self._model is None:
            self.load_model()
        if self._model is None:
            raise RuntimeError("YOLO model could not be initialized.")

        results = self._model.predict(
            source=str(image_path),
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []

        xyxy = boxes.xyxy.tolist()
        classes = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        detections: List[Detection] = []
        for coords, cls_id, conf in zip(xyxy, classes, confidences):
            class_int = int(cls_id)
            if class_int < 0 or class_int >= len(DETECTION_CLASS_NAMES):
                raise ValueError(f"Unexpected class index {class_int} from model output.")
            detections.append(
                Detection(
                    region_class=RegionClass(class_int),
                    confidence=float(conf),
                    bbox=BoundingBox(
                        x_min=float(coords[0]),
                        y_min=float(coords[1]),
                        x_max=float(coords[2]),
                        y_max=float(coords[3]),
                    ),
                )
            )
        return self._dedupe_staff_detections(detections)

    def _dedupe_staff_detections(self, detections: Sequence[Detection]) -> List[Detection]:
        threshold = max(0.0, min(1.0, float(self.config.dedupe_iou_threshold)))
        if threshold <= 0.0:
            return list(detections)
        overlap_threshold = max(0.0, min(1.0, float(self.config.dedupe_vertical_overlap_threshold)))
        center_ratio = max(0.0, float(self.config.dedupe_center_distance_ratio))
        staff = [det for det in detections if det.region_class == RegionClass.STAFF]
        others = [det for det in detections if det.region_class != RegionClass.STAFF]
        if len(staff) < 2:
            return list(detections)

        kept: List[Detection] = []
        for det in sorted(staff, key=lambda item: item.confidence, reverse=True):
            is_duplicate = False
            for prev in kept:
                if det.bbox.iou(prev.bbox) >= threshold:
                    is_duplicate = True
                    break
                overlap_ratio = det.bbox.vertical_overlap_ratio(prev.bbox)
                min_height = max(1e-6, min(det.bbox.height, prev.bbox.height))
                center_distance = abs(det.bbox.y_center - prev.bbox.y_center)
                if overlap_ratio >= overlap_threshold and center_distance <= (center_ratio * min_height):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)
        kept.sort(key=lambda item: item.bbox.y_center)
        return [*kept, *others]

    def _merge_groups_by_brackets(
        self,
        groups: List[List[Detection]],
        brackets: Sequence[Detection],
    ) -> List[List[Detection]]:
        if not groups or not brackets:
            return groups

        parent = list(range(len(groups)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        group_bands: List[BoundingBox] = []
        for group in groups:
            group_bands.append(
                BoundingBox(
                    x_min=min(item.bbox.x_min for item in group),
                    y_min=min(item.bbox.y_min for item in group),
                    x_max=max(item.bbox.x_max for item in group),
                    y_max=max(item.bbox.y_max for item in group),
                )
            )

        for bracket in brackets:
            hit_indices = [
                idx
                for idx, band in enumerate(group_bands)
                if bracket.bbox.y_min <= band.y_center <= bracket.bbox.y_max
            ]
            if len(hit_indices) < 2:
                continue
            base = hit_indices[0]
            for idx in hit_indices[1:]:
                union(base, idx)

        merged: Dict[int, List[Detection]] = {}
        for idx, group in enumerate(groups):
            root = find(idx)
            merged.setdefault(root, []).extend(group)

        merged_groups = list(merged.values())
        for group in merged_groups:
            group.sort(key=lambda item: item.bbox.y_center)
        merged_groups.sort(key=lambda g: min(item.bbox.y_center for item in g))
        return merged_groups

    def group_staff_into_systems(self, detections: Sequence[Detection]) -> List[List[Detection]]:
        staffs = sorted(
            [det for det in detections if det.region_class == RegionClass.STAFF],
            key=lambda det: det.bbox.y_center,
        )
        if not staffs:
            return []
        brackets = [det for det in detections if det.region_class == RegionClass.SYSTEM_BRACKET]

        groups: List[List[Detection]] = []
        for staff in staffs:
            placed = False
            for group in groups:
                band = BoundingBox(
                    x_min=min(item.bbox.x_min for item in group),
                    y_min=min(item.bbox.y_min for item in group),
                    x_max=max(item.bbox.x_max for item in group),
                    y_max=max(item.bbox.y_max for item in group),
                )
                if staff.bbox.overlaps_vertically(band, self.config.system_overlap_threshold):
                    group.append(staff)
                    placed = True
                    break
            if not placed:
                groups.append([staff])

        for group in groups:
            group.sort(key=lambda item: item.bbox.y_center)
        groups.sort(key=lambda g: min(item.bbox.y_center for item in g))

        # Proximity-based merging for grand staff detection:
        # If consecutive single-staff groups are close together (gap < 1.5x staff height)
        # and much farther from the next system, merge them.
        groups = self._merge_groups_by_proximity(groups)

        return self._merge_groups_by_brackets(groups, brackets)

    def _merge_groups_by_proximity(self, groups: List[List["Detection"]]) -> List[List["Detection"]]:
        """Merge consecutive single-staff groups that are close together (grand staff)."""
        if len(groups) < 2:
            return groups

        # Compute gaps between consecutive groups
        gaps: List[float] = []
        for i in range(len(groups) - 1):
            bottom_of_current = max(item.bbox.y_max for item in groups[i])
            top_of_next = min(item.bbox.y_min for item in groups[i + 1])
            gaps.append(top_of_next - bottom_of_current)

        if not gaps:
            return groups

        # Compute typical staff height
        staff_heights = []
        for group in groups:
            for item in group:
                staff_heights.append(item.bbox.height)
        median_height = sorted(staff_heights)[len(staff_heights) // 2] if staff_heights else 100.0

        # Classify gaps: "small" = within-system, "large" = between-system
        # Use a threshold relative to staff height: small gaps are < 0.5x staff height
        proximity_threshold = median_height * 0.5

        # Also use gap clustering: if gaps form two distinct clusters, use that
        sorted_gaps = sorted(gaps)
        if len(sorted_gaps) >= 3:
            # Find the largest jump between consecutive sorted gaps
            max_jump = 0.0
            max_jump_idx = 0
            for i in range(len(sorted_gaps) - 1):
                jump = sorted_gaps[i + 1] - sorted_gaps[i]
                if jump > max_jump:
                    max_jump = jump
                    max_jump_idx = i
            # If there's a clear bimodal split, use the midpoint as threshold
            if max_jump > median_height * 0.3:
                proximity_threshold = (sorted_gaps[max_jump_idx] + sorted_gaps[max_jump_idx + 1]) / 2.0

        # Merge groups where the gap is below threshold
        merged: List[List["Detection"]] = [list(groups[0])]
        for i in range(len(gaps)):
            if gaps[i] < proximity_threshold:
                merged[-1].extend(groups[i + 1])
            else:
                merged.append(list(groups[i + 1]))

        for group in merged:
            group.sort(key=lambda item: item.bbox.y_center)

        return merged

    def crop_staff_regions(
        self,
        image_path: Path,
        detections: Sequence[Detection],
        output_dir: Path,
    ) -> List[StaffCrop]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for crop export. Install with: pip install pillow") from exc

        image = Image.open(image_path)
        width, height = image.size
        systems = self.group_staff_into_systems(detections)
        output_dir.mkdir(parents=True, exist_ok=True)

        crops: List[StaffCrop] = []
        stem = image_path.stem
        safe_stem = "".join(char if (char.isalnum() or char in {"_", "-"}) else "_" for char in stem)
        if not safe_stem:
            safe_stem = "page"

        # First pass: deduplicate staves within each system.
        all_compact_systems: List[List[Detection]] = []
        for system in systems:
            system_compact: List[Detection] = []
            for detection in system:
                if not system_compact:
                    system_compact.append(detection)
                    continue
                prev = system_compact[-1]
                overlap_ratio = detection.bbox.vertical_overlap_ratio(prev.bbox)
                min_height = max(1e-6, min(detection.bbox.height, prev.bbox.height))
                center_distance = abs(detection.bbox.y_center - prev.bbox.y_center)
                if overlap_ratio >= max(0.0, min(1.0, float(self.config.dedupe_vertical_overlap_threshold))) and center_distance <= (
                    max(0.0, float(self.config.dedupe_center_distance_ratio)) * min_height
                ):
                    system_compact[-1] = detection
                else:
                    system_compact.append(detection)
            all_compact_systems.append(system_compact)

        # Build list of (bbox, system_index) for adaptive vertical padding.
        all_staff_entries: List[Tuple[BoundingBox, int]] = []
        for sys_i, compact_sys in enumerate(all_compact_systems):
            for det in compact_sys:
                all_staff_entries.append((det.bbox, sys_i))
        all_staff_entries.sort(key=lambda e: e[0].y_center)

        system_x_bounds: List[Tuple[float, float]] = []
        for system_compact in all_compact_systems:
            if not system_compact:
                system_x_bounds.append((0.0, float(width)))
                continue
            detected_x_min = min(det.bbox.x_min for det in system_compact)
            detected_x_max = max(det.bbox.x_max for det in system_compact)
            system_x_min = 0.0 if bool(self.config.full_width_left_page_edge) else detected_x_min
            system_x_max = float(width) if bool(self.config.full_width_right_page_edge) else detected_x_max
            if system_x_max <= system_x_min:
                system_x_min, system_x_max = 0.0, float(width)
            system_x_bounds.append((system_x_min, system_x_max))

        for system_idx, system_compact in enumerate(all_compact_systems):
            if bool(self.config.merge_system_staves):
                # One crop per system: union all staff bboxes vertically.
                system_y_min = min(det.bbox.y_min for det in system_compact)
                system_y_max = max(det.bbox.y_max for det in system_compact)
                staff_heights_in_sys = [det.bbox.height for det in system_compact]
                median_staff_h = sorted(staff_heights_in_sys)[len(staff_heights_in_sys) // 2]
                desired_v_pad = median_staff_h * self.config.vertical_padding_ratio
                min_v_pad = max(median_staff_h * 0.05, float(self.config.min_vertical_padding_px))

                # Find nearest neighbor gaps above and below, tracking same vs cross-system.
                nearest_above_gap = float("inf")
                nearest_below_gap = float("inf")
                for other_bbox, other_sys_idx in all_staff_entries:
                    if other_sys_idx == system_idx:
                        continue
                    if other_bbox.y_center < system_y_min:
                        gap = system_y_min - other_bbox.y_max
                        if gap < nearest_above_gap:
                            nearest_above_gap = gap
                    elif other_bbox.y_center > system_y_max:
                        gap = other_bbox.y_min - system_y_max
                        if gap < nearest_below_gap:
                            nearest_below_gap = gap

                # Adaptive padding: generous (70%) for same-system neighbors (piano
                # grand staff where content extends between staves), strict (45%)
                # for cross-system neighbors to avoid capturing other systems.
                def _clamp_pad_sys(desired: float, gap: float) -> float:
                    if gap <= 0:
                        return min_v_pad
                    return max(min_v_pad, min(desired, gap * 0.55))

                top_pad = _clamp_pad_sys(desired_v_pad, nearest_above_gap) if nearest_above_gap < float("inf") else desired_v_pad
                bottom_pad = _clamp_pad_sys(desired_v_pad, nearest_below_gap) if nearest_below_gap < float("inf") else desired_v_pad

                padded = BoundingBox(
                    x_min=0.0,
                    y_min=system_y_min - top_pad,
                    x_max=float(width),
                    y_max=system_y_max + bottom_pad,
                ).clip(image_width=width, image_height=height)

                x0, y0, x1, y1 = int(round(padded.x_min)), int(round(padded.y_min)), int(round(padded.x_max)), int(round(padded.y_max))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = image.crop((x0, y0, x1, y1))
                crop_name = f"{safe_stem}__sys{system_idx:02d}.png"
                crop_path = output_dir / crop_name
                try:
                    crop.save(str(crop_path))
                except OSError:
                    crop_path = output_dir / f"{safe_stem}__sys{system_idx:02d}__alt.png"
                    crop.save(str(crop_path))
                crops.append(StaffCrop(source_image=str(image_path), crop_path=str(crop_path), system_index=system_idx, staff_index=0, bbox=padded))
            else:
                for staff_idx, detection in enumerate(system_compact):
                    staff_h = detection.bbox.height
                    desired_v_pad = staff_h * self.config.vertical_padding_ratio
                    left_pad = detection.bbox.width * self.config.horizontal_padding_ratio
                    right_pad = max(
                        detection.bbox.width * self.config.horizontal_padding_ratio,
                        float(self.config.min_right_padding_px),
                    )
                    min_v_pad = max(staff_h * 0.05, float(self.config.min_vertical_padding_px))

                    # Find nearest neighbor gaps above and below, tracking same vs cross-system.
                    nearest_above_gap = float("inf")
                    above_same_system = False
                    nearest_below_gap = float("inf")
                    below_same_system = False
                    for other_bbox, other_sys_idx in all_staff_entries:
                        if other_bbox is detection.bbox:
                            continue
                        if other_bbox.y_center < detection.bbox.y_center:
                            gap = detection.bbox.y_min - other_bbox.y_max
                            if gap < nearest_above_gap:
                                nearest_above_gap = gap
                                above_same_system = (other_sys_idx == system_idx)
                        elif other_bbox.y_center > detection.bbox.y_center:
                            gap = other_bbox.y_min - detection.bbox.y_max
                            if gap < nearest_below_gap:
                                nearest_below_gap = gap
                                below_same_system = (other_sys_idx == system_idx)

                    # Adaptive padding: generous (70%) for same-system neighbors (piano
                    # grand staff where content extends between staves), strict (45%)
                    # for cross-system neighbors to avoid capturing other systems.
                    def _clamp_pad(desired: float, gap: float, same_system: bool) -> float:
                        if gap <= 0:
                            return min_v_pad
                        ratio = 0.70 if same_system else 0.55
                        return max(min_v_pad, min(desired, gap * ratio))

                    top_pad = _clamp_pad(desired_v_pad, nearest_above_gap, above_same_system) if nearest_above_gap < float("inf") else desired_v_pad
                    bottom_pad = _clamp_pad(desired_v_pad, nearest_below_gap, below_same_system) if nearest_below_gap < float("inf") else desired_v_pad

                    padded = BoundingBox(
                        x_min=detection.bbox.x_min - left_pad,
                        y_min=detection.bbox.y_min - top_pad,
                        x_max=detection.bbox.x_max + right_pad,
                        y_max=detection.bbox.y_max + bottom_pad,
                    ).clip(image_width=width, image_height=height)
                    if bool(self.config.enforce_full_width_crops):
                        system_x_min, system_x_max = system_x_bounds[system_idx]
                        padded = BoundingBox(
                            x_min=system_x_min,
                            y_min=padded.y_min,
                            x_max=min(float(width), system_x_max + float(self.config.min_right_padding_px)),
                            y_max=padded.y_max,
                        ).padded(
                            vertical_ratio=0.0,
                            horizontal_ratio=self.config.horizontal_padding_ratio,
                        ).clip(image_width=width, image_height=height)

                    x0, y0, x1, y1 = int(round(padded.x_min)), int(round(padded.y_min)), int(round(padded.x_max)), int(round(padded.y_max))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    crop = image.crop((x0, y0, x1, y1))
                    crop_name = f"{safe_stem}__sys{system_idx:02d}__staff{staff_idx:02d}.png"
                    crop_path = output_dir / crop_name
                    try:
                        crop.save(str(crop_path))
                    except OSError:
                        crop_path = output_dir / f"{safe_stem}__sys{system_idx:02d}__staff{staff_idx:02d}__alt.png"
                        crop.save(str(crop_path))
                    crops.append(StaffCrop(source_image=str(image_path), crop_path=str(crop_path), system_index=system_idx, staff_index=staff_idx, bbox=padded))
        return crops

    def write_crop_manifest(self, crops: Sequence[StaffCrop], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for crop in crops:
                handle.write(
                    json.dumps(
                        {
                            "source_image": crop.source_image,
                            "crop_path": crop.crop_path,
                            "system_index": crop.system_index,
                            "staff_index": crop.staff_index,
                            "bbox": {
                                "x_min": crop.bbox.x_min,
                                "y_min": crop.bbox.y_min,
                                "x_max": crop.bbox.x_max,
                                "y_max": crop.bbox.y_max,
                            },
                        }
                    )
                    + "\n"
                )

    def build_training_data_yaml(
        self,
        page_manifest_path: Path,
        output_dir: Path,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
    ) -> Path:
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")
        if not page_manifest_path.exists():
            raise FileNotFoundError(f"Page manifest not found: {page_manifest_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        split_lists = {"train": [], "val": [], "test": []}

        with page_manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {page_manifest_path}:{line_no}") from exc
                image_path = entry.get("png_path")
                label_path = entry.get("label_path")
                label_valid = bool(entry.get("yolo_label_valid", True))
                if not image_path or not label_path or not label_valid:
                    continue
                sample_id = str(entry.get("page_id", f"row-{line_no}"))
                split = _deterministic_split(
                    sample_id=sample_id,
                    seed=self.config.seed,
                    train=train_ratio,
                    val=val_ratio,
                )
                split_lists[split].append(str(image_path))

        split_file_paths: Dict[str, Path] = {}
        for split_name, items in split_lists.items():
            split_file = output_dir / f"{split_name}.txt"
            split_file.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
            split_file_paths[split_name] = split_file

        data_yaml = output_dir / "data.yaml"
        yaml_root = output_dir.resolve()
        yaml_lines = [
            f"path: '{yaml_root.as_posix()}'",
            "train: train.txt",
            "val: val.txt",
            "test: test.txt",
            "names:",
        ]
        for class_id, class_name in enumerate(DETECTION_CLASS_NAMES):
            yaml_lines.append(f"  {class_id}: {class_name}")
        yaml_text = "\n".join(yaml_lines) + "\n"
        data_yaml.write_text(yaml_text, encoding="utf-8")
        return data_yaml
