#!/usr/bin/env python3
"""Testes rápidos do prepare_road_videos.py usando os vídeos em videos/."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from prepare_road_videos import (
    check_ffmpeg,
    extract_frames,
    find_videos,
    get_video_info,
    process_single_video,
    sample_timestamps,
    score_frame,
    slow_down_video,
    speed_up_video,
)

ROOT = Path(__file__).parent
VIDEOS = ROOT / "videos"


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FALHA] {msg}")
    sys.exit(1)


def test_environment() -> None:
    print("\n=== Ambiente ===")
    if not check_ffmpeg():
        fail("ffmpeg não encontrado no PATH")
    ok("ffmpeg disponível")

    videos = find_videos(VIDEOS)
    if len(videos) < 1:
        fail(f"nenhum vídeo em {VIDEOS}")
    ok(f"{len(videos)} vídeo(s) em videos/")


def test_core_functions(video: Path) -> None:
    print("\n=== Funções principais ===")
    info = get_video_info(video)
    if info is None:
        fail(f"não abriu {video.name}")
    ok(f"get_video_info: {info.duration_sec:.1f}s {info.fps}fps")

    ts = sample_timestamps(info.duration_sec, 5, 2.0, 2.0, 42)
    if len(ts) < 1:
        fail("sample_timestamps retornou vazio")
    ok(f"sample_timestamps: {len(ts)} timestamps")

    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        fail("leitura de frame com OpenCV falhou")
    meta, thumb, reason = score_frame(
        frame, min_brightness=25, max_brightness=235, selected_thumbs=[]
    )
    if reason:
        fail(f"score_frame rejeitou frame válido: {reason}")
    ok(f"score_frame: final={meta.final_score:.3f} sharp={meta.sharpness:.1f}")


def test_sampling_modes(video: Path, tmp: Path) -> None:
    print("\n=== Modos de amostragem ===")
    for mode in ("random", "uniform", "smart", "hybrid"):
        out = tmp / f"frames_{mode}"
        result = extract_frames(
            video_path=video,
            output_dir=out,
            frames_per_video=6,
            skip_start=2.0,
            skip_end=2.0,
            seed=42,
            min_brightness=25,
            max_brightness=235,
            sampling_mode=mode,  # type: ignore[arg-type]
            overwrite=True,
        )
        if len(result.selected) < 1:
            fail(f"modo {mode}: nenhum frame extraído")
        if not all(m.path and Path(m.path).exists() for m in result.selected):
            fail(f"modo {mode}: arquivo de frame ausente")
        ok(
            f"{mode}: {len(result.selected)} frames, "
            f"{len(result.rejected)} rejeitados, "
            f"score médio={sum(m.final_score for m in result.selected)/len(result.selected):.3f}"
        )


def test_video_processing(video: Path, tmp: Path) -> None:
    print("\n=== Processamento de vídeo (ffmpeg) ===")
    slow_out = tmp / "slow.mp4"
    if not slow_down_video(video, slow_out, 1.5, overwrite=True):
        fail("slow_down_video falhou")
    if not slow_out.exists() or slow_out.stat().st_size < 1000:
        fail("arquivo desacelerado inválido")
    ok(f"desacelerar 1.5x: {slow_out.stat().st_size // 1024} KB")

    fast_out = tmp / "fast.mp4"
    if not speed_up_video(video, fast_out, 2.0, overwrite=True):
        fail("speed_up_video falhou")
    ok(f"acelerar 2x: {fast_out.stat().st_size // 1024} KB")


def test_full_pipeline_all_videos() -> None:
    print("\n=== Pipeline completo (todos os vídeos) ===")
    processed = ROOT / "processed_videos"
    frames = ROOT / "frames"
    reports = ROOT / "reports"

    videos = find_videos(VIDEOS)
    results = []

    for video in videos:
        print(f"  → {video.name}")
        r = process_single_video(
            video,
            processed_dir=processed,
            frames_dir=frames,
            reports_dir=reports,
            slow_factor=1.5,
            frames_per_video=10,
            sampling_mode="hybrid",
            overwrite=True,
            do_slow=True,
            do_frames=True,
        )
        report_path = reports / f"{video.stem}_report.json"
        if r.error and r.frames_extracted == 0:
            fail(f"{video.name}: {r.error}")
        if not report_path.exists():
            fail(f"relatório ausente: {report_path}")

        with report_path.open(encoding="utf-8") as fh:
            report = json.load(fh)

        if report.get("sampling_mode") != "hybrid":
            fail("sampling_mode incorreto no JSON")
        if len(report.get("frames", [])) != r.frames_extracted:
            fail("contagem de frames no JSON inconsistente")

        slow_path = processed / f"{video.stem}_slow_1.5x.mp4"
        if not slow_path.exists():
            fail(f"vídeo desacelerado ausente: {slow_path}")

        frame_dir = frames / video.stem
        frame_files = list(frame_dir.glob("frame_*.jpg"))
        if len(frame_files) != r.frames_extracted:
            fail(f"frames em disco ({len(frame_files)}) != extraídos ({r.frames_extracted})")

        results.append(
            (video.name, r.frames_extracted, len(r.rejected_candidates), slow_path.stat().st_size)
        )
        ok(
            f"{video.name}: {r.frames_extracted} frames, "
            f"{len(r.rejected_candidates)} rejeitados, "
            f"slow={slow_path.stat().st_size//1024}KB"
        )

    print(f"\n  Resumo: {len(results)}/{len(videos)} vídeos processados com sucesso")


def main() -> int:
    print("Testes prepare_road_videos.py")
    print(f"Python: {sys.executable}")

    test_environment()
    videos = find_videos(VIDEOS)
    shortest = min(videos, key=lambda p: p.stat().st_size)

    with tempfile.TemporaryDirectory(prefix="road_vid_test_") as tmpdir:
        tmp = Path(tmpdir)
        test_core_functions(shortest)
        test_sampling_modes(shortest, tmp)
        test_video_processing(shortest, tmp)

    test_full_pipeline_all_videos()

    print("\n=== Todos os testes passaram ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
