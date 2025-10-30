import argparse
import os
from pathlib import Path


def write_overfit_yaml(base_yaml_path: Path, out_yaml_path: Path) -> None:
    """Create a YOLO data YAML where val/test both point to train to encourage overfitting."""
    # Minimal parser: read the path and names from the original YAML without full YAML deps
    # Fallback to simple line parsing to avoid adding dependencies.
    base_yaml_text = base_yaml_path.read_text(encoding="utf-8")

    # Extract root path for the dataset (path: ...)
    root = None
    names_block = None
    nc_line = None
    for line in base_yaml_text.splitlines():
        if line.strip().startswith("path:"):
            root = line.split(":", 1)[1].strip()
        elif line.strip().startswith("nc:"):
            nc_line = line
        elif line.strip().startswith("names:"):
            names_block = line

    if not root:
        raise ValueError("Failed to parse 'path' from base YAML. Please provide a standard Ultralytics data YAML.")

    # Compose new YAML text where val/test both reuse train/images
    overfit_yaml = [
        f"path: {root}",
        "train: train/images",
        "val: train/images",
        "test: train/images",
    ]

    if nc_line:
        overfit_yaml.append(nc_line)
    if names_block:
        overfit_yaml.append(names_block)

    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    out_yaml_path.write_text("\n".join(overfit_yaml) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train an intentionally overfitted YOLO detector for oral medications.")
    parser.add_argument(
        "--data-yaml",
        default=r"C:\\Users\\hyeon\\Desktop\\ai05-level1-project\\yolo_dataset_clean\\data_clean.yaml",
        help="Path to the original Ultralytics data YAML (with train/val paths).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model weights to start from (e.g., yolov8m.pt). Defaults to yolov8m.pt if present, else yolo11n.pt in repo root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (higher encourages overfitting).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (tune based on GPU memory).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=True,
        help="Enable mixed precision training (recommended).",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Use half precision (FP16) where supported.",
    )
    parser.add_argument(
        "--project",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "Exp"),
        help="Project directory for Ultralytics runs.",
    )
    parser.add_argument(
        "--name",
        default="overfit_yolo",
        help="Run name inside the project directory.",
    )
    args = parser.parse_args()

    # Lazy import to avoid hard dependency for tools that only inspect the repo
    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        raise SystemExit(
            f"Ultralytics and torch are required to run training. Please install them in your environment.\nError: {e}"
        )

    repo_root = Path(__file__).resolve().parents[1]

    # Decide default model if not provided (prefer heavier if available, fallback handled on OOM)
    if args.model is None:
        y8m = repo_root / "yolov8m.pt"
        y11n = repo_root / "yolo11n.pt"
        if y8m.exists():
            model_path = str(y8m)
        elif y11n.exists():
            model_path = str(y11n)
        else:
            model_path = "yolov8m.pt"  # rely on environment's path
    else:
        model_path = args.model

    # Prepare overfit data YAML
    base_yaml_path = Path(args.data_yaml)
    if not base_yaml_path.exists():
        raise SystemExit(f"Data YAML not found: {base_yaml_path}")

    overfit_yaml_path = repo_root / "config" / "data_overfit.yaml"
    write_overfit_yaml(base_yaml_path, overfit_yaml_path)

    # Device selection
    device = 0 if torch.cuda.is_available() else "cpu"

    # Build training configuration that intentionally minimizes augmentation to overfit
    train_kwargs = dict(
        data=str(overfit_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=device,
        optimizer="AdamW",
        pretrained=True,
        deterministic=True,
        workers=args.workers,
        amp=args.amp,
        half=args.half,
        compile=False,
        # Reduce/disable most augmentations
        mosaic=0.0,
        copy_paste=0.0,
        mixup=0.0,
        erasing=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        scale=0.0,
        shear=0.0,
        translate=0.0,
        perspective=0.0,
        close_mosaic=0,
        # Validation still runs; but val/test now point to train for overfit scoring
        val=True,
        patience=0,  # do not early stop; run full epochs
        verbose=True,
    )

    print("[INFO] Starting training with model:", model_path)
    print("[INFO] Using data YAML:", overfit_yaml_path)
    print("[INFO] Device:", device)

    def attempt_train(weights: str, kwargs: dict) -> tuple[bool, str]:
        try:
            mdl = YOLO(weights)
            mdl.train(**kwargs)
            return True, weights
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[WARN] CUDA out of memory encountered. Adjusting settings and retrying...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False, weights
            raise

    # OOM fallback loop: reduce batch -> reduce imgsz -> switch to smaller model if available
    cur_batch = train_kwargs["batch"]
    cur_imgsz = train_kwargs["imgsz"]
    cur_weights = model_path

    max_attempts = 6
    attempt = 1
    while attempt <= max_attempts:
        print(f"[INFO] Attempt {attempt}/{max_attempts} | weights={Path(cur_weights).name} | batch={cur_batch} | imgsz={cur_imgsz}")
        train_kwargs["batch"] = cur_batch
        train_kwargs["imgsz"] = cur_imgsz
        ok, used_weights = attempt_train(cur_weights, train_kwargs)
        if ok:
            model_path = used_weights
            break
        # Not ok: OOM
        if cur_batch > 1:
            cur_batch = max(1, cur_batch // 2)
        elif cur_imgsz > 512:
            cur_imgsz = cur_imgsz - 128
        else:
            # Switch to a smaller model if possible
            y11n = repo_root / "yolo11n.pt"
            if ("v8m" in Path(cur_weights).name or "yolov8m" in Path(cur_weights).name) and y11n.exists():
                print("[WARN] Switching to yolo11n.pt due to tight memory.")
                cur_weights = str(y11n)
            else:
                print("[ERROR] Unable to fit model even with minimal settings. Consider running on CPU or further reducing image size.")
                raise SystemExit(1)
        attempt += 1

    # Evaluate on "test" split (which also points to train here)
    from ultralytics import YOLO as _YOLO
    final_model = _YOLO(model_path)
    print("[INFO] Validating on test split (overfit: test=train)")
    _ = final_model.val(data=str(overfit_yaml_path), split="test", imgsz=train_kwargs["imgsz"], device=device)

    # Optionally export the model for inference
    try:
        print("[INFO] Exporting model to TorchScript (optional)")
        model.export(format="torchscript")
    except Exception:
        # Export is optional; ignore failures
        pass


if __name__ == "__main__":
    main()
