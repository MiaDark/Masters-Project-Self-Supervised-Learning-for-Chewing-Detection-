"""Zero-ablation on the jointly-trained 80/10/10 models (masked_ssl,
classical_ml), evaluated from the ALREADY-preprocessed
data/interim_normalized_80_10_10 cache -- the same interim data both models
were actually trained on, and the same data compare_bites.py (bite_evaluation.md
Sec 2.1, the dissertation's bout-level headline table) reads -- rather than
reconstructing each recording from its original raw CSV the way
checkpoints/cross_sensor_ablation.md's Round-1 zero-ablation study did
(chewing_ssl.predict_new_recording's --from-split path, --raw-root
data/raw_unscaled/raw_unscaled).

Reuses zero_out_modality/predict_classical_ml/predict_masked_ssl UNCHANGED
from predict_new_recording.py -- the only new code here is (a) population
discovery straight from the interim cache + splits.json instead of
discover_recordings-on-raw-CSVs, and (b) writing the zeroed segment to a
temp parquet, since both reused functions take a seg_path, not a DataFrame
in memory.

Purpose: make the "all channels" row of this ablation agree EXACTLY with
compare_bites.py's headline bout-level table (both now read literally the
same cached, preprocessed data), so the dissertation's bout-level,
Round-1-ablation, and Round-2-retrained tables no longer show three
slightly different "same" numbers for three unrelated reasons.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from chewing_ssl.classical_ml.extract_features import DRIVERS
from chewing_ssl.common.data.windowing import WINDOW_LEN, extract_participant, list_tags
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_dual_mode,
    evaluate_recording,
    postprocess_config_from_args,
)
from chewing_ssl.predict_new_recording import (
    DEFAULT_CLASSICAL_ML_MODEL,
    DEFAULT_FEATURE_MEDIANS,
    DEFAULT_SSL_CHECKPOINT,
    predict_classical_ml,
    predict_masked_ssl,
    zero_out_modality,
)
from chewing_ssl.ssl.masked.evaluate_bouts import load_finetuned_model


def discover_test_segments(interim_root: Path, splits_path: Path) -> list[tuple[str, str, Path]]:
    """Every test-purpose segment parquet already sitting in the interim
    cache -- mirrors segment.py's own dataset/participant discovery (same
    splits.json, same "skip pretrain_val_holdout" rule) so this reproduces
    exactly the population segment.py/extract_features.py (classical ML) and
    ssl/masked/evaluate_bouts.py (SSL) built their own test artefacts from,
    Spectacle_NonMeal included."""
    splits = json.loads(splits_path.read_text())
    datasets = [d for d in splits if d != "pretrain_val_holdout"]
    out = []
    for dataset in datasets:
        if not (interim_root / dataset).exists():
            continue
        for tag in list_tags(interim_root, dataset):
            participant = extract_participant(dataset, tag)
            if splits[dataset].get(participant) != "test":
                continue
            for seg_path in sorted((interim_root / dataset / tag).glob("segment_*.parquet")):
                out.append((dataset, tag, seg_path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized_80_10_10"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed_80_10_10/splits.json"))
    parser.add_argument("--classical-ml-model-path", type=Path, default=DEFAULT_CLASSICAL_ML_MODEL)
    parser.add_argument("--ssl-checkpoint", type=Path, default=DEFAULT_SSL_CHECKPOINT)
    parser.add_argument("--feature-medians", type=Path, default=DEFAULT_FEATURE_MEDIANS)
    parser.add_argument(
        "--ssl-smooth-window-s",
        type=float,
        default=2.0,
        help="Overrides --smooth-window-s for the SSL side only, same default as "
        "compare_bites.py's own --ssl-smooth-window-s.",
    )
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")
    parser.add_argument(
        "--out-csv", type=Path, default=Path("checkpoints/cross_sensor_ablation_cached/summary.csv")
    )
    args = parser.parse_args()

    segments = discover_test_segments(args.interim_root, args.splits)
    n_datasets = len({d for d, _, _ in segments})
    print(f"Discovered {len(segments)} test-purpose segment(s) across {n_datasets} dataset(s)")

    feature_medians = pd.Series(json.loads(args.feature_medians.read_text()))
    ssl_model, ssl_meta = load_finetuned_model(args.ssl_checkpoint)
    ssl_model.eval()
    ssl_window_len = ssl_meta["window_len"]

    # Same base config for both pipelines except smoothing (asymmetric by
    # design, see compare_bites.py's own docstring) -- built once, then
    # cloned per pipeline/condition below via postprocess_config_from_args's
    # dataclass, exactly as compare_bites.py's main() does.
    base_cfg_args = argparse.Namespace(**vars(args))

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_seg_path = Path(tmpdir) / "segment.parquet"

        for model_name in ("masked_ssl", "classical_ml"):
            for only in ("all", "nav", "accel"):
                cfg_args = argparse.Namespace(**vars(base_cfg_args))
                cfg_args.smooth_window_s = args.ssl_smooth_window_s if model_name == "masked_ssl" else 0.0
                # ENVELOPE_CHANNELS is nav-only -- under accel-only the raw
                # envelope would be computed on an all-zero signal, same
                # reasoning/override predict_new_recording.py applies.
                cfg_args.envelope_split = args.envelope_split and only != "accel"
                cfg = postprocess_config_from_args(cfg_args)

                results = []
                for dataset, tag, seg_path in segments:
                    df = pd.read_parquet(seg_path)
                    df = zero_out_modality(df, only)
                    df.to_parquet(tmp_seg_path, index=False)

                    if model_name == "masked_ssl":
                        rec = predict_masked_ssl(tmp_seg_path, ssl_model, ssl_window_len)
                    else:
                        rec = predict_classical_ml(
                            tmp_seg_path,
                            args.classical_ml_model_path,
                            feature_medians,
                            WINDOW_LEN,
                            list(DRIVERS),
                            50.0,
                        )
                    rec["dataset"], rec["tag"], rec["segment"] = dataset, tag, seg_path.stem
                    rec["participant"] = f"{dataset}/{tag}/{seg_path.stem}"
                    results.append(
                        evaluate_recording(
                            rec, cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
                        )
                    )

                dual = aggregate_dual_mode(results)
                print(
                    f"{model_name:14s} only={only:6s}  n_gt={dual['n_gt_bouts']:4.0f}  "
                    f"n_pred={dual['n_pred_bouts']:4.0f}  IoU F1={dual['iou']['f1']:.3f}  "
                    f"tol F1={dual['tolerance']['f1']:.3f}"
                )
                rows.append(
                    {
                        "model": model_name,
                        "only": only,
                        "n_gt_bouts": dual["n_gt_bouts"],
                        "n_pred_bouts": dual["n_pred_bouts"],
                        "iou_f1": dual["iou"]["f1"],
                        "tolerance_f1": dual["tolerance"]["f1"],
                    }
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
