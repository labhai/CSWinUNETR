"""Predict wrinkle masks from masked RGB faces and their weak texture maps."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..datasets.ffhq_wrinkle import index_pngs, input_tensor, load_pair, read_ids
from ..models import CSWinUNETR
from ..utils.metrics import foreground_dice, thin_structure_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument(
        "--ids-file", type=Path, help="Optional subset; use the test split for evaluation"
    )
    parser.add_argument(
        "--mask-dir", type=Path, help="Optional manual masks for per-image Dice evaluation"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ffhq"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Also report clDice, Betti errors and HD95 (requires [eval])",
    )
    args = parser.parse_args()
    if args.all_metrics and args.mask_dir is None:
        parser.error("all-metrics requires mask-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty")
    images, textures = index_pngs(args.image_dir), index_pngs(args.texture_dir)
    masks = index_pngs(args.mask_dir) if args.mask_dir else None
    ids = read_ids(args.ids_file) if args.ids_file else sorted(images)
    if not ids:
        parser.error("No input images found")
    for kind, files in (("image", images), ("texture", textures), ("mask", masks)):
        if files is not None:
            missing = [case for case in ids if case not in files]
            if missing:
                parser.error(f"Missing {kind} files for IDs: {', '.join(missing[:10])}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_args = saved["model_args"]
    if any(
        model_args.get(key) != value
        for key, value in {"in_channels": 4, "out_channels": 2, "spatial_dims": 2}.items()
    ):
        parser.error("This FFHQ command requires a 2D, four-channel, two-class checkpoint")
    model = CSWinUNETR(**model_args)
    model.load_state_dict(saved["model"], strict=True)
    model.to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    device = torch.device(args.device)
    latency_ms = []
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, case in enumerate(ids):
            rgb, texture = load_pair(images[case], textures[case])
            inputs = input_tensor(rgb, texture).unsqueeze(0).to(device)
            if device.type == "cuda":
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
            forward_start = time.perf_counter()
            logits = model(inputs)
            if device.type == "cuda":
                end.record()
                end.synchronize()
                elapsed_ms = begin.elapsed_time(end)
            else:
                elapsed_ms = 1000 * (time.perf_counter() - forward_start)
            if index >= 5:
                latency_ms.append(elapsed_ms)
            prediction = logits.argmax(1).cpu()
            Image.fromarray(prediction[0].numpy().astype(np.uint8)).save(
                args.output_dir / f"{case}.png"
            )
            record = {"id": case}
            if masks is not None:
                with Image.open(masks[case]) as image:
                    target = torch.from_numpy((np.array(image.convert("L")) > 0).copy()).unsqueeze(
                        0
                    )
                if target.shape != prediction.shape:
                    raise ValueError(f"Prediction/mask size mismatch for {case}")
                record["dice"] = foreground_dice(prediction, target).item()
                if args.all_metrics:
                    record.update(thin_structure_metrics(prediction[0].numpy(), target[0].numpy()))
            records.append(record)
    elapsed = time.perf_counter() - started
    performance = {
        "elapsed_seconds": elapsed,
        "images_per_second": len(records) / elapsed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": "float32",
        "batch_size": 1,
        "warmup_cases": min(5, len(ids)),
        "timed_cases": len(latency_ms),
    }
    if latency_ms:
        performance.update(
            {
                "forward_mean_ms": float(np.mean(latency_ms)),
                "forward_median_ms": float(np.median(latency_ms)),
                "forward_p95_ms": float(np.percentile(latency_ms, 95)),
                "forward_images_per_second": 1000 / float(np.mean(latency_ms)),
            }
        )
    if device.type == "cuda":
        performance.update(
            {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "gpu": torch.cuda.get_device_name(device),
            }
        )
    summary = {
        "count": len(records),
        "cases": records,
        "performance": performance,
        "checkpoint_epoch": saved["epoch"],
        "model_args": model_args,
    }
    if masks is not None:
        summary["mean_dice"] = sum(row["dice"] for row in records) / len(records)
        if args.all_metrics:
            for metric in ("cldice", "betti0_error", "betti1_error", "betti_error", "hd95"):
                summary[f"mean_{metric}"] = sum(row[metric] for row in records) / len(records)
    (args.output_dir / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}))


if __name__ == "__main__":
    main()
