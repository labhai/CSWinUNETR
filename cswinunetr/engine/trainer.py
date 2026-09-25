"""Train CSWinUNETR on FFHQ-Wrinkle."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader

from ..datasets import SPLIT_DIR, FFHQWrinkle, validate_splits
from ..models import CSWinUNETR
from ..utils.losses import dice_ce_loss
from ..utils.metrics import foreground_dice


def save_checkpoint(state, path):
    """Atomically replace a checkpoint."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    scores = []
    for batch in loader:
        prediction = model(batch["image"].to(device)).argmax(1)
        scores.extend(foreground_dice(prediction, batch["mask"].to(device)).cpu().tolist())
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ffhq"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--num-heads", type=int, nargs=4)
    parser.add_argument("--stripe-widths", type=int, nargs=4, default=[1, 2, 7, 7])
    parser.add_argument("--pool-ratios", type=int, nargs=4, default=[8, 4, 2, 1])
    parser.add_argument("--strip-kernel-sizes", type=int, nargs=4, default=[9, 7, 5, 3])
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument(
        "--sds-stages",
        type=int,
        nargs=4,
        choices=(0, 1),
        default=[1, 0, 0, 0],
        help="SDSConv flags for encoder stages 1-4 (0=off, 1=on)",
    )
    parser.add_argument("--sds-kernel-sizes", type=int, nargs=4, default=[11, 9, 7, 5])
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--patience", type=int, default=30, help="Validation checks without improvement"
    )
    parser.add_argument("--resume", action="store_true", help="Resume output-dir/last.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Recompute encoder activations to reduce training memory",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; workers must be nonnegative")
    if args.eval_interval < 1 or args.patience < 1:
        parser.error("eval-interval and patience must be positive")
    if args.resume and not (args.output_dir / "last.pt").is_file():
        parser.error("resume requires output-dir/last.pt")
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty to avoid overwriting a previous run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    counts = validate_splits(args.split_dir)
    train = FFHQWrinkle(args.data_root, "train", args.split_dir, args.image_size)
    val = FFHQWrinkle(args.data_root, "val", args.split_dir, args.image_size)
    loader_args = {
        "num_workers": args.workers,
        "worker_init_fn": seed_worker,
        "pin_memory": args.device.startswith("cuda"),
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(val, batch_size=1, **loader_args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    model_args = {
        "in_channels": 4,
        "out_channels": 2,
        "spatial_dims": 2,
        "sds_stages": args.sds_stages,
        "sds_kernel_sizes": args.sds_kernel_sizes,
        "feature_size": args.feature_size,
        "depths": args.depths,
        "num_heads": args.num_heads,
        "stripe_widths": args.stripe_widths,
        "pool_ratios": args.pool_ratios,
        "strip_kernel_sizes": args.strip_kernel_sizes,
        "mlp_ratio": args.mlp_ratio,
        "drop_path_rate": args.drop_path_rate,
    }
    model = CSWinUNETR(**model_args, use_checkpoint=args.use_checkpoint).to(device)
    optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=0.05)
    # EMA tracks training weights, separately from schedule-free evaluation weights.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings["split_counts"] = counts
    split_out = args.output_dir / "splits"
    split_out.mkdir(exist_ok=args.resume)
    for split in ("train", "val", "test"):
        content = (args.split_dir / f"{split}.txt").read_text()
        if args.resume:
            if (split_out / f"{split}.txt").read_text() != content:
                parser.error("Cannot resume with different split IDs")
        else:
            (split_out / f"{split}.txt").write_text(content)
    best, stale, step, start_epoch = -1.0, 0, 0, 1
    if args.resume:
        saved = torch.load(args.output_dir / "last.pt", map_location="cpu", weights_only=True)
        for key in (
            "batch_size",
            "image_size",
            "workers",
            "lr",
            "seed",
            "use_checkpoint",
            "sds_stages",
            "sds_kernel_sizes",
            "feature_size",
            "depths",
            "num_heads",
            "stripe_widths",
            "pool_ratios",
            "strip_kernel_sizes",
            "mlp_ratio",
            "drop_path_rate",
            "eval_interval",
            "patience",
            "data_root",
        ):
            if settings[key] != saved["settings"].get(key, parser.get_default(key)):
                parser.error(f"Cannot resume with a different {key}")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        best, stale, step = saved["best"], saved["stale"], saved["step"]
        start_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"], generator)
        # A crash after logging but before checkpoint replacement may leave extra rows.
        metrics_path = args.output_dir / "metrics.jsonl"
        rows = [
            line
            for line in metrics_path.read_text().splitlines()
            if json.loads(line)["epoch"] < start_epoch
        ]
        metrics_path.write_text("\n".join(rows) + ("\n" if rows else ""))
        del saved
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    environment = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": "float32",
        "model_args": model_args,
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        optimizer.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        timed_steps, timed_samples, compute_seconds = 0, 0, 0.0
        total_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            # Exclude warmup batches from throughput measurements.
            timing = device.type == "cuda" and batch_index >= 5
            if timing:
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
            image, target = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_ce_loss(model(image), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    if step == 0:
                        average.copy_(current)
                    else:
                        average.lerp_(current, 1 - 0.9999)
                for average, current in zip(ema.buffers(), model.buffers()):
                    average.copy_(current)
            step += 1
            if timing:
                end.record()
                end.synchronize()
                compute_seconds += begin.elapsed_time(end) / 1000
                timed_steps += 1
                timed_samples += image.shape[0]
            total_loss += loss.item() * image.shape[0]
        train_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "loss": total_loss / len(train),
            "train_seconds": train_seconds,
            "train_images_per_second": len(train) / train_seconds,
        }
        if device.type == "cuda":
            row.update(
                {
                    "train_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "train_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
            )
        if timed_steps:
            row.update(
                {
                    "timed_steps": timed_steps,
                    "compute_step_ms": 1000 * compute_seconds / timed_steps,
                    "compute_images_per_second": timed_samples / compute_seconds,
                }
            )
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            validation_start = time.perf_counter()
            row["val_dice"] = evaluate(ema, val_loader, device)
            row["validation_seconds"] = time.perf_counter() - validation_start
            if row["val_dice"] > best:
                best, stale = row["val_dice"], 0
                save_checkpoint(
                    {
                        "model": ema.state_dict(),
                        "epoch": epoch,
                        "val_dice": best,
                        "model_args": model_args,
                        "settings": settings,
                    },
                    args.output_dir / "best.pt",
                )
            else:
                stale += 1
        row["epoch_seconds"] = time.perf_counter() - epoch_start
        with (args.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        save_checkpoint(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "step": step,
                "model_args": model_args,
                "settings": settings,
                "rng": rng_state(generator),
            },
            args.output_dir / "last.pt",
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
