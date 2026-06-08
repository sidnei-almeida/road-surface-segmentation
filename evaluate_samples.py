#!/usr/bin/env python3
"""Avalia o modelo em amostras aleatórias de test/valid e gera relatório visual."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATASET = Path("/home/a1rm4x/Downloads/dataset")
MODEL_PATH = ROOT / "models" / "best.pt"
OUTPUT_DIR = ROOT / "metrics" / "evaluation"
REPORT_JSON = OUTPUT_DIR / "evaluation_results.json"

IMGSZ = 768
CONF = 0.25
IOU_THRESH = 0.5
SAMPLES_PER_SPLIT = 8
SEED = 42

# Classes do dataset Roboflow consideradas "pothole" no ground truth
POTHOLE_CLASS_IDS = {1, 2, 4}  # Pothole, Potholes, pothole


@dataclass
class InstanceEval:
    gt_index: int | None
    pred_index: int | None
    iou: float
    verdict: str  # tp, fp, fn


@dataclass
class ImageEval:
    split: str
    image_name: str
    image_path: str
    gt_potholes: int
    pred_potholes: int
    tp: int
    fp: int
    fn: int
    mean_iou: float
    verdict: str  # correct, partial, missed, false_positive, correct_negative
    comparison_path: str
    notes: str


def parse_yolo_seg_label(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """Lê polígonos YOLO segmentation do arquivo de label."""
    instances: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return instances

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        pts = np.stack([coords[0::2] * img_w, coords[1::2] * img_h], axis=1).astype(np.int32)
        instances.append((cls_id, pts))
    return instances


def polygon_to_mask(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.size > 0:
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def match_instances(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> tuple[int, int, int, float]:
    """Greedy matching por IoU. Retorna tp, fp, fn, mean_iou dos matches."""
    if not gt_masks and not pred_masks:
        return 0, 0, 0, 1.0
    if not gt_masks:
        return 0, len(pred_masks), 0, 0.0
    if not pred_masks:
        return 0, 0, len(gt_masks), 0.0

    pairs: list[tuple[float, int, int]] = []
    for gi, gm in enumerate(gt_masks):
        for pi, pm in enumerate(pred_masks):
            iou = mask_iou(gm, pm)
            if iou > 0:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched_ious: list[float] = []

    for iou, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        if iou >= IOU_THRESH:
            used_gt.add(gi)
            used_pred.add(pi)
            matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_masks) - len(used_pred)
    fn = len(gt_masks) - len(used_gt)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return tp, fp, fn, mean_iou


def classify_image(gt_count: int, pred_count: int, tp: int, fp: int, fn: int) -> tuple[str, str]:
    if gt_count == 0 and pred_count == 0:
        return "correct_negative", "Sem buracos no GT e nenhuma detecção — correto."
    if gt_count > 0 and pred_count == 0:
        return "missed", f"Buracos no GT ({gt_count}) não detectados."
    if gt_count == 0 and pred_count > 0:
        return "false_positive", f"{pred_count} detecção(ões) sem buraco anotado no GT."
    if fp == 0 and fn == 0:
        return "correct", f"Todas as {tp} instância(s) detectadas corretamente."
    if tp > 0:
        return "partial", f"Detectou {tp}/{gt_count}, mas com {fp} FP e {fn} FN."
    return "missed", "Predições presentes, porém sem overlap suficiente (IoU < 0.5)."


def draw_comparison(
    image: np.ndarray,
    gt_polygons: list[np.ndarray],
    pred_masks: list[np.ndarray],
    title: str,
) -> np.ndarray:
    vis = image.copy()
    overlay = vis.copy()

    for poly in gt_polygons:
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
    for mask in pred_masks:
        colored = np.zeros_like(overlay)
        colored[mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

    vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

    # Legenda
    h, w = vis.shape[:2]
    bar = np.zeros((56, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.putText(bar, "GT: verde", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    cv2.putText(bar, "Pred: vermelho", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(bar, title[:80], (220, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return np.vstack([vis, bar])


def sample_images(split: str, n: int, rng: random.Random) -> list[Path]:
    img_dir = DATASET / split / "images"
    images = sorted(img_dir.glob("*"))
    rng.shuffle(images)
    return images[:n]


def evaluate_image(model: YOLO, image_path: Path, split: str) -> ImageEval:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Falha ao ler {image_path}")

    h, w = image.shape[:2]
    label_path = DATASET / split / "labels" / f"{image_path.stem}.txt"
    instances = parse_yolo_seg_label(label_path, w, h)

    gt_polygons = [pts for cls_id, pts in instances if cls_id in POTHOLE_CLASS_IDS]
    gt_masks = [polygon_to_mask(p, (h, w)) for p in gt_polygons]

    results = model.predict(image_path, imgsz=IMGSZ, conf=CONF, verbose=False)
    pred_masks: list[np.ndarray] = []
    if results and results[0].masks is not None:
        for mask_tensor in results[0].masks.data:
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_masks.append((mask > 0.5).astype(np.uint8))

    tp, fp, fn, mean_iou = match_instances(gt_masks, pred_masks)
    verdict, notes = classify_image(len(gt_masks), len(pred_masks), tp, fp, fn)

    title = f"{split} | GT:{len(gt_masks)} Pred:{len(pred_masks)} | {verdict.upper()}"
    comparison = draw_comparison(image, gt_polygons, pred_masks, title)

    out_name = f"{split}_{image_path.stem}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return ImageEval(
        split=split,
        image_name=image_path.name,
        image_path=str(image_path),
        gt_potholes=len(gt_masks),
        pred_potholes=len(pred_masks),
        tp=tp,
        fp=fp,
        fn=fn,
        mean_iou=round(mean_iou, 3),
        verdict=verdict,
        comparison_path=str(out_path.relative_to(ROOT)),
        notes=notes,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    model = YOLO(str(MODEL_PATH))

    evaluations: list[ImageEval] = []
    for split in ("valid", "test"):
        for img_path in sample_images(split, SAMPLES_PER_SPLIT, rng):
            evaluations.append(evaluate_image(model, img_path, split))

    summary = {
        "model": str(MODEL_PATH),
        "dataset": str(DATASET),
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou_threshold": IOU_THRESH,
        "samples_per_split": SAMPLES_PER_SPLIT,
        "total_images": len(evaluations),
        "verdict_counts": {},
        "aggregate": {},
        "images": [asdict(e) for e in evaluations],
    }

    for v in ("correct", "partial", "missed", "false_positive", "correct_negative"):
        summary["verdict_counts"][v] = sum(1 for e in evaluations if e.verdict == v)

    summary["aggregate"] = {
        "total_tp": sum(e.tp for e in evaluations),
        "total_fp": sum(e.fp for e in evaluations),
        "total_fn": sum(e.fn for e in evaluations),
        "accuracy_rate": round(
            sum(1 for e in evaluations if e.verdict in ("correct", "correct_negative"))
            / len(evaluations),
            3,
        ),
        "detection_rate": round(
            sum(1 for e in evaluations if e.verdict in ("correct", "partial"))
            / max(1, sum(1 for e in evaluations if e.gt_potholes > 0)),
            3,
        ),
    }

    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Avaliadas {len(evaluations)} imagens")
    print("Vereditos:", summary["verdict_counts"])
    print("JSON:", REPORT_JSON)


if __name__ == "__main__":
    main()
