from __future__ import annotations

import argparse
import json

from .common.constants import DEFAULT_ARTIFACTS, DEFAULT_SOURCE_ARTIFACTS


def parser():
    root = argparse.ArgumentParser(description="過去5か月＋現在値による12か月最大加速度予測")
    commands = root.add_subparsers(dest="command", required=True)

    retrieval = commands.add_parser("build-retrieval", help="12か月有効ガイドDBを構築")
    retrieval.add_argument("--source-artifact-dir", default=str(DEFAULT_SOURCE_ARTIFACTS))
    retrieval.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "retrieval"))
    retrieval.add_argument("--min-valid-months", type=int, default=8)
    retrieval.add_argument("--no-progress", action="store_true")

    prepare = commands.add_parser("prepare-datasets", help="6時点入力・12か月Residualデータを構築")
    prepare.add_argument("--source-artifact-dir", default=str(DEFAULT_SOURCE_ARTIFACTS))
    prepare.add_argument("--retrieval-dir", default=str(DEFAULT_ARTIFACTS / "retrieval"))
    prepare.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    prepare.add_argument("--device")
    prepare.add_argument("--max-train", type=int)
    prepare.add_argument("--max-validation", type=int)
    prepare.add_argument("--max-inference", type=int)
    prepare.add_argument("--no-progress", action="store_true")

    prepare_absolute = commands.add_parser("prepare-absolute", help="Cross-Attention absolute-value datasets")
    prepare_absolute.add_argument("--source-artifact-dir", default=str(DEFAULT_SOURCE_ARTIFACTS))
    prepare_absolute.add_argument("--retrieval-dir", default=str(DEFAULT_ARTIFACTS / "retrieval"))
    prepare_absolute.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "datasets_absolute_attention"))
    prepare_absolute.add_argument("--device")
    prepare_absolute.add_argument("--max-train", type=int); prepare_absolute.add_argument("--max-validation", type=int)
    prepare_absolute.add_argument("--max-inference", type=int); prepare_absolute.add_argument("--no-progress", action="store_true")

    train_parser = commands.add_parser("train", help="AttentionなしResidual U-Netを学習")
    train_parser.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    train_parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "model"))
    train_parser.add_argument("--device")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--no-resume", action="store_true")
    train_parser.add_argument("--no-progress", action="store_true")

    validation = commands.add_parser("validate", help="validationを100系列で評価")
    validation.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    validation.add_argument("--checkpoint", default=str(DEFAULT_ARTIFACTS / "model" / "best_model.pt"))
    validation.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "validation"))
    validation.add_argument("--device")
    validation.add_argument("--num-samples", type=int, default=100)
    validation.add_argument("--sampling-steps", type=int, default=50)
    validation.add_argument("--max-records", type=int)
    validation.add_argument("--no-progress", action="store_true")
    validation.add_argument("--initial-noise-scale", type=float, default=1.0)

    selection = commands.add_parser("select-sampling", help="Compare nine DDIM sampling configurations")
    selection.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets_absolute_attention"))
    selection.add_argument("--checkpoint", default=str(DEFAULT_ARTIFACTS / "models_absolute_attention" / "unet" / "best_model.pt"))
    selection.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "validation_absolute_attention"))
    selection.add_argument("--device"); selection.add_argument("--num-samples", type=int, default=100)
    selection.add_argument("--max-records", type=int); selection.add_argument("--no-progress", action="store_true")

    prediction = commands.add_parser("predict", help="inferenceをDDIMで正式予測")
    prediction.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    prediction.add_argument("--checkpoint", default=str(DEFAULT_ARTIFACTS / "model" / "best_model.pt"))
    prediction.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "predictions"))
    prediction.add_argument("--device")
    prediction.add_argument("--num-samples", type=int, default=100)
    prediction.add_argument("--sampling-steps", type=int, default=50)
    prediction.add_argument("--max-records", type=int)
    prediction.add_argument("--no-save-samples", action="store_true")
    prediction.add_argument("--no-progress", action="store_true")
    prediction.add_argument("--initial-noise-scale", type=float, default=1.0)

    evaluation = commands.add_parser("evaluate", help="正解分離後に予測を評価")
    evaluation.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    evaluation.add_argument("--prediction-dir", default=str(DEFAULT_ARTIFACTS / "predictions"))
    evaluation.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "evaluation"))
    evaluation.add_argument("--bootstrap", type=int, default=1000)
    evaluation.add_argument("--plot", action="store_true")
    evaluation.add_argument("--plot-max-targets", type=int, default=100)
    evaluation.add_argument("--y-max", type=float, default=6.0)
    evaluation.add_argument("--dpi", type=int, default=150)
    evaluation.add_argument("--no-progress", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "build-retrieval":
        from .retrieval.build import build_retrieval_database
        result = build_retrieval_database(
            args.source_artifact_dir, args.output_dir,
            min_valid_months=args.min_valid_months, progress=not args.no_progress,
        )
    elif args.command in {"prepare-datasets", "prepare-absolute"}:
        from .datasets.build import prepare_datasets
        result = prepare_datasets(
            args.source_artifact_dir, args.retrieval_dir, args.output_dir, device=args.device,
            max_train=args.max_train, max_validation=args.max_validation,
            max_inference=args.max_inference, progress=not args.no_progress,
            target_mode="absolute" if args.command == "prepare-absolute" else "residual",
        )
    elif args.command == "train":
        from .training.train import train
        result = train(
            args.dataset_dir, args.output_dir, device=args.device, epochs=args.epochs,
            batch_size=args.batch_size, resume=not args.no_resume, progress=not args.no_progress,
        )
    elif args.command == "validate":
        from .inference.validate import validate
        result = validate(
            args.dataset_dir, args.checkpoint, args.output_dir, device=args.device,
            num_samples=args.num_samples, sampling_steps=args.sampling_steps,
            max_records=args.max_records, progress=not args.no_progress,
            initial_noise_scale=args.initial_noise_scale,
        )
    elif args.command == "select-sampling":
        from .inference.select_sampling import select_sampling
        result = select_sampling(
            args.dataset_dir, args.checkpoint, args.output_dir, device=args.device,
            num_samples=args.num_samples, max_records=args.max_records,
            progress=not args.no_progress,
        )
    elif args.command == "predict":
        from .inference.predict import predict
        result = predict(
            args.dataset_dir, args.checkpoint, args.output_dir, device=args.device,
            num_samples=args.num_samples, sampling_steps=args.sampling_steps,
            save_samples=not args.no_save_samples, max_records=args.max_records,
            progress=not args.no_progress,
            initial_noise_scale=args.initial_noise_scale,
        )
    else:
        from .evaluation.evaluate import evaluate
        result = evaluate(
            args.dataset_dir, args.prediction_dir, args.output_dir,
            bootstrap=args.bootstrap, plot=args.plot, plot_max_targets=args.plot_max_targets,
            y_max=args.y_max, dpi=args.dpi, progress=not args.no_progress,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
