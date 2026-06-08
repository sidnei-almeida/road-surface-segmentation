#!/usr/bin/env python3
"""
Pré-processamento de vídeos POV/dashcam para dataset de segmentação de danos em vias.

Instalação:
    pip install opencv-python numpy rich textual

Requisito de sistema:
    ffmpeg deve estar instalado e disponível no PATH.
    Exemplo (Debian/Ubuntu): sudo apt install ffmpeg
    Exemplo (Arch): sudo pacman -S ffmpeg
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
MIN_TIMESTAMP_GAP_SEC = 0.5
MAX_BRIGHTNESS_RETRIES = 8
DUPLICATE_RECENT_WINDOW = 15

SamplingMode = Literal["random", "uniform", "smart", "hybrid"]

# ROI da via: faixa inferior ~55%, com margem lateral para ignorar bordas extremas
ROAD_ROI_TOP_FRAC = 0.45
ROAD_ROI_SIDE_MARGIN = 0.05

# Limiares internos de qualidade
MIN_SHARPNESS_VAR = 40.0
DUPLICATE_SIMILARITY_THRESHOLD = 0.92
THUMB_SIZE = (32, 32)
HYBRID_CANDIDATES_PER_SEGMENT = 10
SMART_CANDIDATE_MULTIPLIER = 4

# Padrão de extração densa (~1 frame a cada 0.06 s de vídeo útil)
DEFAULT_FRAME_INTERVAL_SEC = 0.06
DEFAULT_FRAMES_PER_VIDEO = 600


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class VideoInfo:
    path: Path
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class FrameMetadata:
    timestamp_sec: float
    frame_index: int
    path: str | None = None
    brightness: float = 0.0
    sharpness: float = 0.0
    contrast: float = 0.0
    texture: float = 0.0
    duplicate_penalty: float = 0.0
    final_score: float = 0.0
    rejection_reason: str | None = None
    selected: bool = False


@dataclass
class ExtractFramesResult:
    selected: list[FrameMetadata] = field(default_factory=list)
    rejected: list[FrameMetadata] = field(default_factory=list)


@dataclass
class ProcessResult:
    original_name: str
    duration_sec: float
    fps: float
    resolution: str
    slow_factor: float | None = None
    speed_factor: float | None = None
    processed_video_path: str | None = None
    sampling_mode: str | None = None
    frames_requested: int = 0
    frames_extracted: int = 0
    frame_timestamps: list[float] = field(default_factory=list)
    frame_paths: list[str] = field(default_factory=list)
    frame_metadata: list[FrameMetadata] = field(default_factory=list)
    rejected_candidates: list[FrameMetadata] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def ensure_dirs(*paths: Path) -> None:
    """Cria pastas de saída se ainda não existirem."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> bool:
    """Verifica se ffmpeg está instalado."""
    return shutil.which("ffmpeg") is not None


def find_videos(input_dir: Path) -> list[Path]:
    """Lista vídeos suportados na pasta de entrada, ordenados por nome."""
    if not input_dir.exists():
        logger.warning("Pasta de entrada não encontrada: %s", input_dir)
        return []

    videos = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


def get_video_info(video_path: Path) -> VideoInfo | None:
    """Obtém metadados do vídeo via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Não foi possível abrir o vídeo: %s", video_path)
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = frame_count / fps if fps > 0 else 0.0

        return VideoInfo(
            path=video_path,
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
    finally:
        cap.release()


def _run_ffmpeg(cmd: list[str]) -> bool:
    """Executa comando ffmpeg e registra erros."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            logger.error("ffmpeg falhou: %s", stderr[-500:] if stderr else "sem detalhes")
            return False
        return True
    except FileNotFoundError:
        logger.error(
            "ffmpeg não encontrado. Instale-o no sistema e garanta que está no PATH."
        )
        return False


def slow_down_video(
    input_path: Path,
    output_path: Path,
    slow_factor: float,
    overwrite: bool = False,
) -> bool:
    """Gera versão desacelerada do vídeo com ffmpeg."""
    if output_path.exists() and not overwrite:
        logger.info("Vídeo desacelerado já existe, pulando: %s", output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-an",
        "-filter:v",
        f"setpts={slow_factor}*PTS",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        str(output_path),
    ]
    return _run_ffmpeg(cmd)


def speed_up_video(
    input_path: Path,
    output_path: Path,
    speed_factor: float,
    overwrite: bool = False,
) -> bool:
    """Gera versão acelerada do vídeo com ffmpeg."""
    if output_path.exists() and not overwrite:
        logger.info("Vídeo acelerado já existe, pulando: %s", output_path)
        return True

    if speed_factor <= 0:
        logger.error("Fator de aceleração deve ser maior que zero.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pts_multiplier = 1.0 / speed_factor
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-an",
        "-filter:v",
        f"setpts={pts_multiplier}*PTS",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        str(output_path),
    ]
    return _run_ffmpeg(cmd)


def sample_timestamps(
    duration_sec: float,
    frames_per_video: int,
    skip_start: float,
    skip_end: float,
    seed: int,
    min_gap: float = MIN_TIMESTAMP_GAP_SEC,
) -> list[float]:
    """
    Gera timestamps aleatórios distribuídos ao longo do vídeo,
    evitando concentração e mantendo distância mínima entre amostras.
    """
    valid_start = skip_start
    valid_end = duration_sec - skip_end

    if valid_end <= valid_start:
        logger.warning(
            "Intervalo válido vazio (duração=%.2fs, skip_start=%.2f, skip_end=%.2f)",
            duration_sec,
            skip_start,
            skip_end,
        )
        return []

    valid_range = valid_end - valid_start
    rng = random.Random(seed)

    if frames_per_video <= 0:
        return []

    if frames_per_video == 1:
        return [valid_start + valid_range / 2]

    segment_size = valid_range / frames_per_video
    effective_min_gap = min(min_gap, segment_size * 0.4)

    candidates: list[float] = []
    for i in range(frames_per_video):
        seg_start = valid_start + i * segment_size
        seg_end = seg_start + segment_size
        jitter_low = seg_start + segment_size * 0.15
        jitter_high = seg_end - segment_size * 0.15
        if jitter_high <= jitter_low:
            ts = (seg_start + seg_end) / 2
        else:
            ts = rng.uniform(jitter_low, jitter_high)
        candidates.append(ts)

    candidates.sort()

    selected: list[float] = []
    for ts in candidates:
        if all(abs(ts - prev) >= effective_min_gap for prev in selected):
            selected.append(ts)
        else:
            shifted = ts
            for direction in (1, -1):
                for step in range(1, 6):
                    candidate = ts + direction * step * (effective_min_gap * 0.5)
                    if valid_start <= candidate <= valid_end:
                        if all(abs(candidate - prev) >= effective_min_gap for prev in selected):
                            shifted = candidate
                            break
                else:
                    continue
                break
            if all(abs(shifted - prev) >= effective_min_gap for prev in selected):
                selected.append(shifted)

    return sorted(selected)


def sample_timestamps_uniform(
    duration_sec: float,
    frames_per_video: int,
    skip_start: float,
    skip_end: float,
) -> list[float]:
    """Gera timestamps uniformemente espaçados no intervalo válido."""
    valid_start = skip_start
    valid_end = duration_sec - skip_end
    if valid_end <= valid_start or frames_per_video <= 0:
        return []

    if frames_per_video == 1:
        return [valid_start + (valid_end - valid_start) / 2]

    step = (valid_end - valid_start) / (frames_per_video - 1)
    return [valid_start + i * step for i in range(frames_per_video)]


def _valid_time_range(duration_sec: float, skip_start: float, skip_end: float) -> tuple[float, float]:
    return skip_start, duration_sec - skip_end


def resolve_frames_per_video(
    duration_sec: float,
    skip_start: float,
    skip_end: float,
    frames_per_video: int,
    frame_interval: float | None,
) -> int:
    """
    Define quantos frames extrair por vídeo.
    Se frame_interval for informado, calcula pela duração útil.
    """
    valid_start, valid_end = _valid_time_range(duration_sec, skip_start, skip_end)
    valid_duration = valid_end - valid_start
    if valid_duration <= 0:
        return 0

    if frame_interval is not None and frame_interval > 0:
        return max(1, int(valid_duration / frame_interval))

    if frames_per_video > 0:
        return frames_per_video

    return max(1, int(valid_duration / DEFAULT_FRAME_INTERVAL_SEC))


def _sampling_density(frames_per_video: int, valid_duration: float) -> float:
    if valid_duration <= 0:
        return 0.0
    return frames_per_video / valid_duration


def _duplicate_settings(
    frames_per_video: int, valid_duration: float
) -> tuple[float, int]:
    """
    Retorna (limiar, janela) para detecção de duplicatas.
    Extração densa só rejeita frames quase idênticos ao imediato anterior.
    """
    density = _sampling_density(frames_per_video, valid_duration)
    if density >= 10:
        return 0.999, 2
    if density >= 5:
        return 0.996, 4
    if density >= 2:
        return 0.98, 8
    return DUPLICATE_SIMILARITY_THRESHOLD, DUPLICATE_RECENT_WINDOW


def _effective_sampling_mode(
    sampling_mode: SamplingMode, frames_per_video: int, valid_duration: float
) -> SamplingMode:
    """Em extração densa, hybrid vira uniforme com filtro de qualidade (mais estável)."""
    if sampling_mode == "hybrid" and _sampling_density(frames_per_video, valid_duration) >= 4:
        return "uniform"
    return sampling_mode


def _candidates_per_segment(frames_per_video: int) -> int:
    return max(8, min(30, frames_per_video // 25 + 8))


def _recent_thumbs(thumbs: list[np.ndarray], window: int) -> list[np.ndarray]:
    if window <= 0 or not thumbs:
        return []
    if len(thumbs) <= window:
        return thumbs
    return thumbs[-window:]


def _preload_frames(
    cap: cv2.VideoCapture,
    timestamps: list[float],
    fps: float,
) -> dict[float, tuple[np.ndarray, int]]:
    """Lê frames em ordem temporal (muito mais rápido que seek repetido)."""
    if not timestamps:
        return {}

    unique_ts = sorted(set(timestamps))
    loaded: dict[float, tuple[np.ndarray, int]] = {}

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, unique_ts[0] * 1000.0 - 50.0))
    target_idx = 0
    frame_counter = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

    while target_idx < len(unique_ts):
        target_ts = unique_ts[target_idx]
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        current_ts = frame_counter / fps if fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_counter += 1

        while target_idx < len(unique_ts) and current_ts >= unique_ts[target_idx]:
            loaded[unique_ts[target_idx]] = (frame.copy(), int(round(unique_ts[target_idx] * fps)))
            target_idx += 1

    return loaded


def extract_road_roi(frame: np.ndarray) -> np.ndarray:
    """Recorta a região inferior da via (~55%), ignorando bordas laterais extremas."""
    h, w = frame.shape[:2]
    y0 = int(h * ROAD_ROI_TOP_FRAC)
    x0 = int(w * ROAD_ROI_SIDE_MARGIN)
    x1 = int(w * (1.0 - ROAD_ROI_SIDE_MARGIN))
    return frame[y0:h, x0:x1]


def _roi_thumbnail(roi_gray: np.ndarray) -> np.ndarray:
    """Miniatura normalizada para comparação de duplicatas."""
    thumb = cv2.resize(roi_gray, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return thumb.astype(np.float32) / 255.0


def _frame_similarity(thumb_a: np.ndarray, thumb_b: np.ndarray) -> float:
    """Similaridade 0-1 entre miniaturas (1 = quase idêntico)."""
    diff = np.mean(np.abs(thumb_a - thumb_b))
    return float(max(0.0, 1.0 - diff))


def _brightness_score(brightness: float, min_b: float, max_b: float) -> float:
    """Pontua brilho ideal em torno do centro do intervalo aceitável."""
    if brightness < min_b or brightness > max_b:
        return 0.0
    center = (min_b + max_b) / 2.0
    half_range = max((max_b - min_b) / 2.0, 1.0)
    distance = abs(brightness - center) / half_range
    return float(max(0.0, 1.0 - distance))


def _normalize_metric(value: float, cap: float) -> float:
    return float(min(1.0, max(0.0, value / cap)))


def score_frame(
    frame: np.ndarray,
    *,
    min_brightness: float,
    max_brightness: float,
    selected_thumbs: list[np.ndarray],
    duplicate_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    duplicate_window: int = DUPLICATE_RECENT_WINDOW,
    check_duplicate: bool = True,
) -> tuple[FrameMetadata, np.ndarray | None, str | None]:
    """
    Calcula métricas de qualidade na ROI da via.
    Retorna metadados, miniatura da ROI e motivo de rejeição (se houver).
    """
    roi = extract_road_roi(frame)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(roi_gray))
    sharpness = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
    contrast = float(np.std(roi_gray))

    edges = cv2.Canny(roi_gray, 50, 150)
    texture = float(np.count_nonzero(edges) / edges.size)

    thumb = _roi_thumbnail(roi_gray)
    duplicate_penalty = 0.0
    recent = _recent_thumbs(selected_thumbs, duplicate_window)
    if check_duplicate and recent:
        duplicate_penalty = max(_frame_similarity(thumb, prev) for prev in recent)

    meta = FrameMetadata(
        timestamp_sec=0.0,
        frame_index=-1,
        brightness=brightness,
        sharpness=sharpness,
        contrast=contrast,
        texture=texture,
        duplicate_penalty=duplicate_penalty,
    )

    if brightness < min_brightness:
        meta.rejection_reason = "too_dark"
        return meta, None, meta.rejection_reason
    if brightness > max_brightness:
        meta.rejection_reason = "too_bright"
        return meta, None, meta.rejection_reason
    if sharpness < MIN_SHARPNESS_VAR:
        meta.rejection_reason = "too_blurry"
        return meta, None, meta.rejection_reason
    if check_duplicate and duplicate_penalty >= duplicate_threshold:
        meta.rejection_reason = "too_similar"
        return meta, None, meta.rejection_reason

    brightness_component = _brightness_score(brightness, min_brightness, max_brightness)
    sharpness_component = _normalize_metric(sharpness, 500.0)
    contrast_component = _normalize_metric(contrast, 80.0)
    texture_component = _normalize_metric(texture, 0.15)

    meta.final_score = float(
        0.20 * brightness_component
        + 0.35 * sharpness_component
        + 0.20 * contrast_component
        + 0.20 * texture_component
        - 0.25 * duplicate_penalty
    )
    return meta, thumb, None


def _read_frame_at_timestamp(
    cap: cv2.VideoCapture,
    timestamp_sec: float,
    fps: float,
) -> tuple[np.ndarray | None, int]:
    """Posiciona o vídeo em um timestamp e lê o frame com índice estimado."""
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    frame_index = int(round(timestamp_sec * fps)) if fps > 0 else -1
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, frame_index
    return frame, frame_index


def _generate_segment_candidates(
    seg_start: float,
    seg_end: float,
    count: int,
    rng: random.Random,
) -> list[float]:
    """Gera timestamps candidatos dentro de um segmento temporal."""
    if seg_end <= seg_start or count <= 0:
        return []
    if count == 1:
        return [(seg_start + seg_end) / 2]

    margin = (seg_end - seg_start) * 0.08
    low = seg_start + margin
    high = seg_end - margin
    if high <= low:
        low, high = seg_start, seg_end

    return [rng.uniform(low, high) for _ in range(count)]


def _pick_hybrid_winner(
    scored: list[tuple[float, FrameMetadata, np.ndarray, np.ndarray]],
    rng: random.Random,
) -> tuple[FrameMetadata, np.ndarray, np.ndarray] | None:
    """Escolhe o melhor candidato com aleatoriedade controlada entre os top-N."""
    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    top_k = min(3, len(scored))
    weights = [0.5, 0.3, 0.2][:top_k]
    total = sum(weights)
    weights = [w / total for w in weights]
    choice = rng.choices(range(top_k), weights=weights, k=1)[0]
    return scored[choice][1], scored[choice][2], scored[choice][3]


def _build_candidate_timestamps(
    sampling_mode: SamplingMode,
    duration_sec: float,
    frames_per_video: int,
    skip_start: float,
    skip_end: float,
    seed: int,
) -> list[list[float]]:
    """
    Retorna listas de timestamps candidatos.
    Para hybrid/smart: uma lista por slot final; para random/uniform: lista única.
    """
    valid_start, valid_end = _valid_time_range(duration_sec, skip_start, skip_end)
    if valid_end <= valid_start or frames_per_video <= 0:
        return []

    rng = random.Random(seed)

    if sampling_mode == "uniform":
        return [sample_timestamps_uniform(duration_sec, frames_per_video, skip_start, skip_end)]

    if sampling_mode == "random":
        return [sample_timestamps(duration_sec, frames_per_video, skip_start, skip_end, seed)]

    if sampling_mode == "hybrid":
        segment_size = (valid_end - valid_start) / frames_per_video
        groups: list[list[float]] = []
        for i in range(frames_per_video):
            seg_start = valid_start + i * segment_size
            seg_end = seg_start + segment_size
            groups.append(
                _generate_segment_candidates(
                    seg_start,
                    seg_end,
                    _candidates_per_segment(frames_per_video),
                    rng,
                )
            )
        return groups

    # smart: muitos candidatos distribuídos, seleção global por score
    multiplier = max(SMART_CANDIDATE_MULTIPLIER, min(12, frames_per_video // 50 + 2))
    total_candidates = max(frames_per_video * multiplier, frames_per_video + 4)
    segment_size = (valid_end - valid_start) / total_candidates
    candidates: list[float] = []
    for i in range(total_candidates):
        seg_start = valid_start + i * segment_size
        seg_end = seg_start + segment_size
        picked = _generate_segment_candidates(seg_start, seg_end, 1, rng)
        if picked:
            candidates.append(picked[0])
    return [candidates]


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
    skip_start: float,
    skip_end: float,
    seed: int,
    min_brightness: float,
    max_brightness: float,
    sampling_mode: SamplingMode = "hybrid",
    overwrite: bool = False,
) -> ExtractFramesResult:
    """
    Extrai frames com amostragem random/uniform/smart/hybrid e pontuação de qualidade.
    """
    empty = ExtractFramesResult()
    info = get_video_info(video_path)
    if info is None:
        return empty

    output_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = sorted(output_dir.glob("frame_*.jpg"))
        if len(existing) >= frames_per_video:
            logger.info(
                "Frames já existem em %s (%d arquivos), pulando extração.",
                output_dir,
                len(existing),
            )
            selected: list[FrameMetadata] = []
            for frame_path in existing[:frames_per_video]:
                try:
                    frame_idx = int(frame_path.stem.split("_")[1])
                except (IndexError, ValueError):
                    frame_idx = -1
                ts = frame_idx / info.fps if info.fps > 0 else 0.0
                selected.append(
                    FrameMetadata(
                        timestamp_sec=ts,
                        frame_index=frame_idx,
                        path=str(frame_path),
                        selected=True,
                    )
                )
            return ExtractFramesResult(selected=selected)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("OpenCV não conseguiu abrir: %s", video_path)
        return empty

    rng = random.Random(seed + 1)
    valid_start, valid_end = _valid_time_range(info.duration_sec, skip_start, skip_end)
    valid_duration = valid_end - valid_start
    dup_threshold, dup_window = _duplicate_settings(frames_per_video, valid_duration)
    effective_mode = _effective_sampling_mode(sampling_mode, frames_per_video, valid_duration)
    candidate_groups = _build_candidate_timestamps(
        effective_mode,
        info.duration_sec,
        frames_per_video,
        skip_start,
        skip_end,
        seed,
    )

    selected: list[FrameMetadata] = []
    rejected: list[FrameMetadata] = []
    selected_thumbs: list[np.ndarray] = []

    mode_label = effective_mode
    if effective_mode != sampling_mode:
        mode_label = f"{sampling_mode}->{effective_mode}"

    logger.info(
        "  Extração: %d frames alvo | modo=%s | dup_threshold=%.3f | janela=%d",
        frames_per_video,
        mode_label,
        dup_threshold,
        dup_window,
    )

    try:
        if effective_mode in ("random", "uniform"):
            base_timestamps = candidate_groups[0] if candidate_groups else []

            def _attempt_timestamps(base_ts: float) -> list[float]:
                attempts = [base_ts]
                for step in range(1, MAX_BRIGHTNESS_RETRIES + 1):
                    for sign in (-1, 1):
                        candidate_ts = base_ts + sign * step * 0.15
                        if valid_start <= candidate_ts <= valid_end:
                            attempts.append(candidate_ts)
                if effective_mode == "random":
                    rng.shuffle(attempts[1:])
                return attempts

            preload_ts: list[float] = []
            for base_ts in base_timestamps:
                preload_ts.extend(_attempt_timestamps(base_ts))
            frame_cache = _preload_frames(cap, preload_ts, info.fps)

            for slot, base_ts in enumerate(base_timestamps, start=1):
                attempts = _attempt_timestamps(base_ts)

                saved = False
                for attempt_ts in attempts:
                    cached = frame_cache.get(attempt_ts)
                    if cached is not None:
                        frame, frame_index = cached
                    else:
                        frame, frame_index = _read_frame_at_timestamp(cap, attempt_ts, info.fps)
                    if frame is None:
                        rejected.append(
                            FrameMetadata(
                                timestamp_sec=attempt_ts,
                                frame_index=frame_index,
                                rejection_reason="read_failed",
                                selected=False,
                            )
                        )
                        continue

                    meta, thumb, reason = score_frame(
                        frame,
                        min_brightness=min_brightness,
                        max_brightness=max_brightness,
                        selected_thumbs=selected_thumbs,
                        duplicate_threshold=dup_threshold,
                        duplicate_window=dup_window,
                    )
                    meta.timestamp_sec = attempt_ts
                    meta.frame_index = frame_index

                    if reason and reason != "too_similar":
                        rejected.append(meta)
                        continue

                    if reason == "too_similar":
                        rejected.append(meta)
                        continue

                    frame_path = output_dir / f"frame_{slot:06d}.jpg"
                    if cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        meta.path = str(frame_path)
                        meta.selected = True
                        selected.append(meta)
                        if thumb is not None:
                            selected_thumbs.append(thumb)
                        saved = True
                        break

                    rejected.append(
                        FrameMetadata(
                            timestamp_sec=attempt_ts,
                            frame_index=frame_index,
                            rejection_reason="write_failed",
                            selected=False,
                        )
                    )

                if not saved:
                    # Fallback: salva o centro do slot mesmo com qualidade marginal
                    fallback_ts = base_ts
                    cached = frame_cache.get(fallback_ts)
                    frame = cached[0] if cached else _read_frame_at_timestamp(cap, fallback_ts, info.fps)[0]
                    if frame is not None:
                        meta, thumb, _ = score_frame(
                            frame,
                            min_brightness=min_brightness,
                            max_brightness=max_brightness,
                            selected_thumbs=selected_thumbs,
                            duplicate_threshold=dup_threshold,
                            duplicate_window=dup_window,
                            check_duplicate=False,
                        )
                        meta.timestamp_sec = fallback_ts
                        meta.frame_index = cached[1] if cached else int(round(fallback_ts * info.fps))
                        frame_path = output_dir / f"frame_{slot:06d}.jpg"
                        if cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            meta.path = str(frame_path)
                            meta.selected = True
                            meta.rejection_reason = "fallback_accepted"
                            selected.append(meta)
                            if thumb is not None:
                                selected_thumbs.append(thumb)
                            saved = True

                if not saved:
                    logger.warning(
                        "Não foi possível salvar frame %d de %s (timestamp base %.2fs)",
                        slot,
                        video_path.name,
                        base_ts,
                    )

        elif effective_mode == "hybrid":
            for slot, timestamps in enumerate(candidate_groups, start=1):
                scored_candidates: list[tuple[float, FrameMetadata, np.ndarray, np.ndarray]] = []

                for attempt_ts in timestamps:
                    frame, frame_index = _read_frame_at_timestamp(cap, attempt_ts, info.fps)
                    if frame is None:
                        rejected.append(
                            FrameMetadata(
                                timestamp_sec=attempt_ts,
                                frame_index=frame_index,
                                rejection_reason="read_failed",
                                selected=False,
                            )
                        )
                        continue

                    meta, thumb, reason = score_frame(
                        frame,
                        min_brightness=min_brightness,
                        max_brightness=max_brightness,
                        selected_thumbs=selected_thumbs,
                        duplicate_threshold=dup_threshold,
                        duplicate_window=dup_window,
                    )
                    meta.timestamp_sec = attempt_ts
                    meta.frame_index = frame_index

                    if reason:
                        rejected.append(meta)
                        continue
                    if thumb is None:
                        continue

                    jitter = rng.uniform(-0.03, 0.03)
                    adjusted_score = meta.final_score + jitter
                    scored_candidates.append((adjusted_score, meta, thumb, frame))

                winner = _pick_hybrid_winner(
                    [(s, m, t, f) for s, m, t, f in scored_candidates],
                    rng,
                )
                if winner is None and timestamps:
                    center_ts = timestamps[len(timestamps) // 2]
                    frame, frame_index = _read_frame_at_timestamp(cap, center_ts, info.fps)
                    if frame is not None:
                        meta, thumb, _ = score_frame(
                            frame,
                            min_brightness=min_brightness,
                            max_brightness=max_brightness,
                            selected_thumbs=selected_thumbs,
                            duplicate_threshold=dup_threshold,
                            duplicate_window=dup_window,
                            check_duplicate=False,
                        )
                        meta.timestamp_sec = center_ts
                        meta.frame_index = frame_index
                        meta.rejection_reason = "fallback_accepted"
                        if thumb is not None:
                            winner = (meta, thumb, frame)

                if winner is None:
                    continue

                meta, thumb, frame = winner
                frame_path = output_dir / f"frame_{slot:06d}.jpg"
                if cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    meta.path = str(frame_path)
                    meta.selected = True
                    selected.append(meta)
                    selected_thumbs.append(thumb)
                else:
                    meta.rejection_reason = "write_failed"
                    rejected.append(meta)

        else:  # smart
            all_candidates = candidate_groups[0] if candidate_groups else []
            evaluated: list[tuple[float, FrameMetadata, np.ndarray, np.ndarray]] = []

            for attempt_ts in all_candidates:
                frame, frame_index = _read_frame_at_timestamp(cap, attempt_ts, info.fps)
                if frame is None:
                    rejected.append(
                        FrameMetadata(
                            timestamp_sec=attempt_ts,
                            frame_index=frame_index,
                            rejection_reason="read_failed",
                            selected=False,
                        )
                    )
                    continue

                meta, thumb, reason = score_frame(
                    frame,
                    min_brightness=min_brightness,
                    max_brightness=max_brightness,
                    selected_thumbs=[],
                    duplicate_threshold=dup_threshold,
                    duplicate_window=dup_window,
                )
                meta.timestamp_sec = attempt_ts
                meta.frame_index = frame_index

                if reason:
                    rejected.append(meta)
                    continue
                if thumb is None:
                    continue

                evaluated.append((meta.final_score, meta, thumb, frame))

            evaluated.sort(key=lambda item: item[0], reverse=True)

            while len(selected) < frames_per_video and evaluated:
                picked = False
                for idx, (score, meta, thumb, frame) in enumerate(evaluated):
                    duplicate_penalty = 0.0
                    if selected_thumbs:
                        duplicate_penalty = max(
                            _frame_similarity(thumb, prev) for prev in selected_thumbs
                        )
                    if duplicate_penalty >= dup_threshold:
                        meta.duplicate_penalty = duplicate_penalty
                        meta.rejection_reason = "too_similar"
                        rejected.append(meta)
                        evaluated.pop(idx)
                        picked = True
                        break

                    frame_path = output_dir / f"frame_{len(selected) + 1:06d}.jpg"
                    if cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        meta.path = str(frame_path)
                        meta.selected = True
                        meta.duplicate_penalty = duplicate_penalty
                        meta.final_score = score
                        selected.append(meta)
                        selected_thumbs.append(thumb)
                    else:
                        meta.rejection_reason = "write_failed"
                        rejected.append(meta)
                    evaluated.pop(idx)
                    picked = True
                    break

                if not picked:
                    break

    finally:
        cap.release()

    selected.sort(key=lambda item: item.timestamp_sec)
    return ExtractFramesResult(selected=selected, rejected=rejected)


def _metadata_to_dict(meta: FrameMetadata) -> dict:
    return {
        "timestamp_sec": meta.timestamp_sec,
        "frame_index": meta.frame_index,
        "path": meta.path,
        "brightness": round(meta.brightness, 2),
        "sharpness": round(meta.sharpness, 2),
        "contrast": round(meta.contrast, 2),
        "texture": round(meta.texture, 4),
        "duplicate_penalty": round(meta.duplicate_penalty, 4),
        "final_score": round(meta.final_score, 4),
        "rejection_reason": meta.rejection_reason,
        "selected": meta.selected,
    }


def write_report(report_path: Path, result: ProcessResult) -> None:
    """Salva relatório JSON do processamento de um vídeo."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "original_file": result.original_name,
        "duration_sec": result.duration_sec,
        "fps": result.fps,
        "resolution": result.resolution,
        "slow_factor": result.slow_factor,
        "speed_factor": result.speed_factor,
        "processed_video_path": result.processed_video_path,
        "sampling_mode": result.sampling_mode,
        "frames_requested": result.frames_requested,
        "frames_extracted": result.frames_extracted,
        "frame_timestamps_sec": result.frame_timestamps,
        "frame_paths": result.frame_paths,
        "frames": [_metadata_to_dict(m) for m in result.frame_metadata],
        "rejected_candidates": [_metadata_to_dict(m) for m in result.rejected_candidates],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if result.error:
        payload["error"] = result.error

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _log_video_info(info: VideoInfo) -> None:
    logger.info("Vídeo: %s", info.path.name)
    logger.info("  Duração : %.2f s", info.duration_sec)
    logger.info("  FPS     : %.2f", info.fps)
    logger.info("  Resolução: %dx%d", info.width, info.height)


def process_single_video(
    video_path: Path,
    *,
    processed_dir: Path,
    frames_dir: Path,
    reports_dir: Path,
    slow_factor: float | None = None,
    speed_factor: float | None = None,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
    frame_interval: float | None = None,
    skip_start: float = 2.0,
    skip_end: float = 2.0,
    seed: int = 42,
    min_brightness: float = 25.0,
    max_brightness: float = 235.0,
    sampling_mode: SamplingMode = "hybrid",
    overwrite: bool = False,
    do_slow: bool = True,
    do_speed: bool = False,
    do_frames: bool = True,
) -> ProcessResult:
    """Processa um único vídeo conforme as opções selecionadas."""
    info = get_video_info(video_path)
    if info is None:
        return ProcessResult(
            original_name=video_path.name,
            duration_sec=0.0,
            fps=0.0,
            resolution="unknown",
            error="Falha ao abrir vídeo com OpenCV",
        )

    _log_video_info(info)
    stem = video_path.stem
    result = ProcessResult(
        original_name=video_path.name,
        duration_sec=info.duration_sec,
        fps=info.fps,
        resolution=f"{info.width}x{info.height}",
        sampling_mode=sampling_mode if do_frames else None,
    )

    source_for_frames = video_path
    processed_path: Path | None = None

    if do_slow and slow_factor is not None:
        result.slow_factor = slow_factor
        processed_path = processed_dir / f"{stem}_slow_{slow_factor}x.mp4"
        if processed_path.exists() and not overwrite:
            result.processed_video_path = str(processed_path)
            source_for_frames = processed_path
            logger.info("  Usando vídeo desacelerado existente: %s", processed_path)
        else:
            ok = slow_down_video(video_path, processed_path, slow_factor, overwrite)
            if ok:
                result.processed_video_path = str(processed_path)
                source_for_frames = processed_path
                logger.info("  Salvo   : %s", processed_path)
            else:
                result.error = "Falha ao desacelerar vídeo com ffmpeg"
                logger.warning("  Falha ao desacelerar, usando vídeo original para frames.")

    if do_speed and speed_factor is not None:
        result.speed_factor = speed_factor
        processed_path = processed_dir / f"{stem}_fast_{speed_factor}x.mp4"
        ok = speed_up_video(video_path, processed_path, speed_factor, overwrite)
        if ok:
            result.processed_video_path = str(processed_path)
            source_for_frames = processed_path
            logger.info("  Salvo   : %s", processed_path)
        else:
            result.error = "Falha ao acelerar vídeo com ffmpeg"
            logger.warning("  Falha ao acelerar, usando vídeo original para frames.")

    target_frames = 0
    if do_frames:
        target_frames = resolve_frames_per_video(
            info.duration_sec,
            skip_start,
            skip_end,
            frames_per_video,
            frame_interval,
        )
        result.frames_requested = target_frames

    if do_frames and target_frames > 0:
        frame_output_dir = frames_dir / stem
        extraction = extract_frames(
            video_path=source_for_frames,
            output_dir=frame_output_dir,
            frames_per_video=target_frames,
            skip_start=skip_start,
            skip_end=skip_end,
            seed=seed,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            sampling_mode=sampling_mode,
            overwrite=overwrite,
        )
        result.frames_extracted = len(extraction.selected)
        result.frame_timestamps = [m.timestamp_sec for m in extraction.selected]
        result.frame_paths = [m.path for m in extraction.selected if m.path]
        result.frame_metadata = extraction.selected
        result.rejected_candidates = extraction.rejected
        logger.info(
            "  Frames  : %d extraídos (%s) em %s | %d candidatos rejeitados",
            len(extraction.selected),
            sampling_mode,
            frame_output_dir,
            len(extraction.rejected),
        )

    report_path = reports_dir / f"{stem}_report.json"
    write_report(report_path, result)
    logger.info("  Relatório: %s", report_path)

    return result


def process_all_videos(args: argparse.Namespace) -> int:
    """Pipeline principal para todos os vídeos da pasta de entrada."""
    input_dir = Path(args.input_dir)
    processed_dir = Path(args.processed_dir)
    frames_dir = Path(args.frames_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dirs(processed_dir, frames_dir, reports_dir)

    needs_ffmpeg = (not args.no_slow) or args.speed_factor is not None
    if needs_ffmpeg and not check_ffmpeg():
        logger.error(
            "ffmpeg não está instalado ou não está no PATH. "
            "Instale-o antes de continuar."
        )
        return 1

    videos = find_videos(input_dir)
    if not videos:
        logger.warning("Nenhum vídeo encontrado em: %s", input_dir)
        return 0

    logger.info("Encontrados %d vídeo(s) em %s", len(videos), input_dir)

    failures = 0
    for video_path in videos:
        logger.info("-" * 60)
        try:
            result = process_single_video(
                video_path,
                processed_dir=processed_dir,
                frames_dir=frames_dir,
                reports_dir=reports_dir,
                slow_factor=args.slow_factor if not args.no_slow else None,
                speed_factor=args.speed_factor,
                frames_per_video=args.frames_per_video,
                frame_interval=args.frame_interval,
                skip_start=args.skip_start,
                skip_end=args.skip_end,
                seed=args.seed,
                min_brightness=args.min_brightness,
                max_brightness=args.max_brightness,
                sampling_mode=args.sampling_mode,
                overwrite=args.overwrite,
                do_slow=not args.no_slow and args.speed_factor is None,
                do_speed=args.speed_factor is not None,
                do_frames=not args.no_frames,
            )
            if result.error:
                failures += 1
        except Exception as exc:
            failures += 1
            logger.exception("Erro inesperado ao processar %s: %s", video_path.name, exc)

    logger.info("=" * 60)
    logger.info(
        "Concluído: %d vídeo(s) processado(s), %d falha(s).",
        len(videos) - failures,
        failures,
    )
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# TUI (Textual)
# ---------------------------------------------------------------------------


def run_tui() -> int:
    """Inicia interface interativa no terminal."""
    try:
        from textual import on, work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container, Horizontal, Vertical, VerticalScroll
        from textual.screen import Screen
        from textual.widgets import (
            Button,
            Footer,
            Header,
            Input,
            Label,
            RichLog,
            Static,
            Switch,
        )
    except ImportError:
        logger.error(
            "TUI requer 'textual' e 'rich'. Instale com:\n"
            "  pip install rich textual"
        )
        return 1

    from rich.panel import Panel
    from rich.text import Text

    class ConfigScreen(Screen):
        """Tela de configuração antes do processamento."""

        BINDINGS = [Binding("escape", "app.pop_screen", "Voltar")]

        def __init__(
            self,
            mode: str,
            defaults: dict,
            callback,
        ) -> None:
            super().__init__()
            self.mode = mode
            self.defaults = defaults
            self.callback = callback

        def compose(self) -> ComposeResult:
            title = {
                "slow": "Desacelerar vídeos",
                "fast": "Acelerar vídeos",
                "frames": "Extrair frames aleatórios",
                "full": "Pipeline completo (desacelerar + frames)",
            }.get(self.mode, "Configurar")

            yield Header()
            yield Static(f"[bold cyan]{title}[/bold cyan]", id="cfg-title")
            with VerticalScroll(id="cfg-form"):
                if self.mode in ("slow", "full"):
                    yield Label("Fator de desaceleração (ex: 1.5 = 50% mais lento):")
                    yield Input(
                        value=str(self.defaults.get("slow_factor", 1.5)),
                        id="slow-factor",
                        type="number",
                    )
                if self.mode == "fast":
                    yield Label("Fator de aceleração (ex: 2.0 = 2x mais rápido):")
                    yield Input(
                        value=str(self.defaults.get("speed_factor", 2.0)),
                        id="speed-factor",
                        type="number",
                    )
                if self.mode in ("frames", "full"):
                    yield Label("Frames por vídeo:")
                    yield Input(
                        value=str(self.defaults.get("frames_per_video", DEFAULT_FRAMES_PER_VIDEO)),
                        id="frames-per-video",
                        type="number",
                    )
                    yield Label("Intervalo entre frames em segundos (0 = usar contagem fixa):")
                    yield Input(
                        value=str(self.defaults.get("frame_interval") or ""),
                        id="frame-interval",
                        type="number",
                    )
                    yield Label("Pular início (segundos):")
                    yield Input(
                        value=str(self.defaults.get("skip_start", 2.0)),
                        id="skip-start",
                        type="number",
                    )
                    yield Label("Pular fim (segundos):")
                    yield Input(
                        value=str(self.defaults.get("skip_end", 2.0)),
                        id="skip-end",
                        type="number",
                    )
                    yield Label("Seed (reprodutibilidade):")
                    yield Input(
                        value=str(self.defaults.get("seed", 42)),
                        id="seed",
                        type="number",
                    )
                    yield Label("Modo de amostragem (random/uniform/smart/hybrid):")
                    yield Input(
                        value=str(self.defaults.get("sampling_mode", "hybrid")),
                        id="sampling-mode",
                    )
                yield Label("Sobrescrever arquivos existentes:")
                with Horizontal():
                    yield Switch(value=self.defaults.get("overwrite", False), id="overwrite")
                    yield Static("Ativado = substitui saídas anteriores")
            with Horizontal(id="cfg-actions"):
                yield Button("Executar", variant="primary", id="run-btn")
                yield Button("Cancelar", id="cancel-btn")
            yield Footer()

        @on(Button.Pressed, "#run-btn")
        def on_run(self) -> None:
            config: dict = {"overwrite": self.query_one("#overwrite", Switch).value}

            try:
                if self.mode in ("slow", "full"):
                    config["slow_factor"] = float(
                        self.query_one("#slow-factor", Input).value
                    )
                if self.mode == "fast":
                    config["speed_factor"] = float(
                        self.query_one("#speed-factor", Input).value
                    )
                if self.mode in ("frames", "full"):
                    config["frames_per_video"] = int(
                        self.query_one("#frames-per-video", Input).value
                    )
                    interval_raw = self.query_one("#frame-interval", Input).value.strip()
                    config["frame_interval"] = float(interval_raw) if interval_raw else None
                    config["skip_start"] = float(self.query_one("#skip-start", Input).value)
                    config["skip_end"] = float(self.query_one("#skip-end", Input).value)
                    config["seed"] = int(self.query_one("#seed", Input).value)
                    mode = self.query_one("#sampling-mode", Input).value.strip().lower()
                    if mode not in ("random", "uniform", "smart", "hybrid"):
                        self.app.notify(
                            "Modo inválido. Use: random, uniform, smart ou hybrid.",
                            severity="error",
                        )
                        return
                    config["sampling_mode"] = mode
            except ValueError:
                self.app.notify("Valores inválidos nos campos numéricos.", severity="error")
                return

            self.app.pop_screen()
            self.callback(self.mode, config)

        @on(Button.Pressed, "#cancel-btn")
        def on_cancel(self) -> None:
            self.app.pop_screen()

    class RoadVideoApp(App):
        """TUI principal para pré-processamento de vídeos."""

        CSS = """
        Screen {
            background: #0d1117;
        }

        #main-container {
            padding: 1 2;
            height: 100%;
        }

        #hero {
            width: 100%;
            height: auto;
            padding: 1 0;
            margin-bottom: 1;
        }

        #video-count {
            color: #8b949e;
            margin-bottom: 1;
        }

        #actions {
            height: auto;
            margin: 1 0;
        }

        #actions Button {
            margin: 0 1 1 0;
            min-width: 28;
        }

        #log-panel {
            border: tall #30363d;
            height: 1fr;
            margin-top: 1;
        }

        RichLog {
            height: 100%;
            background: #161b22;
            border: none;
        }

        #cfg-form {
            height: auto;
            max-height: 20;
            margin: 1 2;
        }

        #cfg-form Label {
            margin-top: 1;
            color: #c9d1d9;
        }

        #cfg-form Input {
            margin-bottom: 1;
        }

        #cfg-actions {
            margin: 1 2;
            height: auto;
        }

        #cfg-actions Button {
            margin-right: 1;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Sair"),
            Binding("1", "mode_slow", "Desacelerar"),
            Binding("2", "mode_fast", "Acelerar"),
            Binding("3", "mode_frames", "Frames"),
            Binding("4", "mode_full", "Completo"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.input_dir = Path("videos")
            self.processed_dir = Path("processed_videos")
            self.frames_dir = Path("frames")
            self.reports_dir = Path("reports")
            self.defaults = {
                "slow_factor": 1.5,
                "speed_factor": 2.0,
                "frames_per_video": DEFAULT_FRAMES_PER_VIDEO,
                "frame_interval": None,
                "skip_start": 2.0,
                "skip_end": 2.0,
                "seed": 42,
                "sampling_mode": "hybrid",
                "overwrite": False,
            }

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Container(id="main-container"):
                hero = Text()
                hero.append("Road Video Preprocessor\n", style="bold white")
                hero.append("Dataset de segmentação — danos em vias", style="italic #58a6ff")
                yield Static(Panel(hero, border_style="#30363d"), id="hero")

                video_count = len(find_videos(self.input_dir))
                yield Static(
                    f"Pasta: {self.input_dir}/  |  {video_count} vídeo(s) encontrado(s)",
                    id="video-count",
                )

                with Horizontal(id="actions"):
                    yield Button("1  Desacelerar", variant="primary", id="btn-slow")
                    yield Button("2  Acelerar", variant="warning", id="btn-fast")
                    yield Button("3  Extrair frames", variant="success", id="btn-frames")
                    yield Button("4  Pipeline completo", id="btn-full")

                yield Static("[bold]Log de execução[/bold]", id="log-title")
                yield RichLog(highlight=True, markup=True, id="log-panel", wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            ensure_dirs(
                self.input_dir,
                self.processed_dir,
                self.frames_dir,
                self.reports_dir,
            )
            log = self.query_one("#log-panel", RichLog)
            log.write("[green]Pastas verificadas/criadas com sucesso.[/green]")
            log.write(f"Entrada: [cyan]{self.input_dir.resolve()}[/cyan]")
            log.write(f"Saída vídeos: [cyan]{self.processed_dir.resolve()}[/cyan]")
            log.write(f"Frames: [cyan]{self.frames_dir.resolve()}[/cyan]")
            log.write(f"Relatórios: [cyan]{self.reports_dir.resolve()}[/cyan]")

            if not check_ffmpeg():
                log.write(
                    "[yellow]Aviso: ffmpeg não encontrado. "
                    "Operações de vídeo falharão até instalá-lo.[/yellow]"
                )

        def _open_config(self, mode: str) -> None:
            self.push_screen(
                ConfigScreen(
                    mode=mode,
                    defaults=self.defaults,
                    callback=self._start_processing,
                )
            )

        def action_mode_slow(self) -> None:
            self._open_config("slow")

        def action_mode_fast(self) -> None:
            self._open_config("fast")

        def action_mode_frames(self) -> None:
            self._open_config("frames")

        def action_mode_full(self) -> None:
            self._open_config("full")

        @on(Button.Pressed, "#btn-slow")
        def btn_slow(self) -> None:
            self._open_config("slow")

        @on(Button.Pressed, "#btn-fast")
        def btn_fast(self) -> None:
            self._open_config("fast")

        @on(Button.Pressed, "#btn-frames")
        def btn_frames(self) -> None:
            self._open_config("frames")

        @on(Button.Pressed, "#btn-full")
        def btn_full(self) -> None:
            self._open_config("full")

        def _start_processing(self, mode: str, config: dict) -> None:
            self.run_worker(
                self._process_worker(mode, config),
                thread=True,
                exclusive=True,
            )

        @work(thread=True, exclusive=True)
        def _process_worker(self, mode: str, config: dict) -> None:
            log = self.query_one("#log-panel", RichLog)

            def write(msg: str) -> None:
                self.call_from_thread(log.write, msg)

            ensure_dirs(
                self.processed_dir,
                self.frames_dir,
                self.reports_dir,
            )

            videos = find_videos(self.input_dir)
            if not videos:
                write("[yellow]Nenhum vídeo encontrado na pasta videos/[/yellow]")
                return

            needs_ffmpeg = mode in ("slow", "fast", "full")
            if needs_ffmpeg and not check_ffmpeg():
                write("[red]ffmpeg não instalado. Abortando.[/red]")
                return

            write(f"[bold]Iniciando modo: {mode}[/bold] ({len(videos)} vídeo(s))")

            failures = 0
            for video_path in videos:
                write(f"\n[cyan]▶ {video_path.name}[/cyan]")
                try:
                    kwargs = {
                        "processed_dir": self.processed_dir,
                        "frames_dir": self.frames_dir,
                        "reports_dir": self.reports_dir,
                        "overwrite": config.get("overwrite", False),
                        "skip_start": config.get("skip_start", 2.0),
                        "skip_end": config.get("skip_end", 2.0),
                        "seed": config.get("seed", 42),
                        "sampling_mode": config.get("sampling_mode", "hybrid"),
                        "frame_interval": config.get("frame_interval"),
                    }

                    if mode == "slow":
                        kwargs.update(
                            slow_factor=config.get("slow_factor", 1.5),
                            do_slow=True,
                            do_frames=False,
                            frames_per_video=0,
                        )
                    elif mode == "fast":
                        kwargs.update(
                            speed_factor=config.get("speed_factor", 2.0),
                            do_speed=True,
                            do_frames=False,
                            frames_per_video=0,
                        )
                    elif mode == "frames":
                        kwargs.update(
                            do_slow=False,
                            do_frames=True,
                            frames_per_video=config.get("frames_per_video", 60),
                        )
                    elif mode == "full":
                        kwargs.update(
                            slow_factor=config.get("slow_factor", 1.5),
                            do_slow=True,
                            do_frames=True,
                            frames_per_video=config.get("frames_per_video", 60),
                        )

                    result = process_single_video(video_path, **kwargs)

                    write(
                        f"  Duração: {result.duration_sec:.1f}s | "
                        f"FPS: {result.fps:.1f} | {result.resolution}"
                    )
                    if result.processed_video_path:
                        write(f"  [green]Vídeo salvo:[/green] {result.processed_video_path}")
                    if result.frames_extracted:
                        write(f"  [green]Frames:[/green] {result.frames_extracted} extraídos")
                    if result.error:
                        write(f"  [yellow]Aviso:[/yellow] {result.error}")
                        failures += 1
                except Exception as exc:
                    failures += 1
                    write(f"  [red]Erro:[/red] {exc}")

            ok = len(videos) - failures
            write(
                f"\n[bold green]Finalizado:[/bold green] "
                f"{ok}/{len(videos)} com sucesso."
            )
            if failures:
                write(f"[yellow]{failures} vídeo(s) com problemas.[/yellow]")

    RoadVideoApp().run()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pré-processa vídeos POV/dashcam para anotação de segmentação.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-dir", default="videos", help="Pasta com vídeos de entrada")
    parser.add_argument(
        "--processed-dir",
        default="processed_videos",
        help="Pasta para vídeos processados",
    )
    parser.add_argument("--frames-dir", default="frames", help="Pasta base para frames")
    parser.add_argument("--reports-dir", default="reports", help="Pasta para relatórios JSON")
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=1.5,
        help="Fator de desaceleração (setpts=N*PTS)",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=None,
        help="Fator de aceleração (ex: 2.0 = 2x mais rápido). Exclui desaceleração.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help="Quantidade de frames por vídeo (padrão alto para datasets grandes)",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help=(
            "Intervalo alvo entre frames em segundos; se definido, "
            f"substitui --frames-per-video (ex: 0.06 ≈ {int(1/0.06)} fps de amostragem)"
        ),
    )
    parser.add_argument(
        "--skip-start",
        type=float,
        default=2.0,
        help="Ignorar os primeiros N segundos na extração de frames",
    )
    parser.add_argument(
        "--skip-end",
        type=float,
        default=2.0,
        help="Ignorar os últimos N segundos na extração de frames",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para amostragem de frames")
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "uniform", "smart", "hybrid"],
        default="hybrid",
        help="Estratégia de amostragem de frames para anotação",
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=25.0,
        help="Brilho mínimo aceitável (0-255)",
    )
    parser.add_argument(
        "--max-brightness",
        type=float,
        default=235.0,
        help="Brilho máximo aceitável (0-255)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescrever arquivos de saída existentes",
    )
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="Não desacelerar vídeos (extrair frames do original)",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Apenas processar vídeo, sem extrair frames",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Abrir interface interativa no terminal",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Sem argumentos extras além do script → abre TUI
    if len(sys.argv) == 1 or args.tui:
        return run_tui()

    return process_all_videos(args)


if __name__ == "__main__":
    sys.exit(main())
