"""Compare saved partition-failure evidence without rerunning inference."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(2**31 - 1)
DEFAULT_VIYT = ROOT / "results/clustering_v2_stage_analysis/stage_counts.csv"
DEFAULT_SYNTHETIC = ROOT / "results/synthetic_vietnamese_benchmark/experiments/holdout/recording_results.csv"
DEFAULT_NATURAL = ROOT / "results/natural_vietnamese_benchmark/full_evaluation/recording_results.csv"
DEFAULT_OUTPUT = ROOT / "results/partition_quality_stability/cross_dataset"


def _read(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    return None if value in (None, "") else float(value)


def _initial_k(row):
    raw = row.get("ahc_diagnostics") or row.get("clustering_diagnostics")
    if not raw:
        return None
    diagnostics = json.loads(raw)
    ahc = diagnostics.get("ahc", diagnostics)
    value = ahc.get("estimated_n_speakers")
    if value is not None:
        return int(value)
    labels = ahc.get("labels")
    return None if labels is None else len(set(labels))


def _reference_k(row):
    value = row.get("reference_speaker_count") or row.get("reference_speakers")
    return None if value in (None, "") else int(value)


def _summarize_benchmark(dataset, path, overlap_condition=None):
    rows = [row for row in _read(path) if row.get("mode") == "full_pipeline"]
    if overlap_condition is not None:
        rows = [row for row in rows if row.get("overlap_condition") == overlap_condition]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration"]].append(row)
    if not grouped:
        raise ValueError(f"{path}: no matching full_pipeline rows")

    output = []
    for configuration, method_rows in sorted(grouped.items()):
        initial_comparable = [(row, _initial_k(row), _reference_k(row)) for row in method_rows]
        initial_comparable = [item for item in initial_comparable
                              if item[1] is not None and item[2] is not None]
        partition_comparable = [
            (row, initial, reference, _number(row.get("ARI")))
            for row, initial, reference in initial_comparable if _number(row.get("ARI")) is not None
        ]
        final_comparable = [(int(row["predicted_speakers"]), _reference_k(row))
                            for row in method_rows
                            if row.get("predicted_speakers") not in (None, "")
                            and _reference_k(row) is not None]
        wrong_partition = [row for row, initial, reference, ari in partition_comparable
                           if initial == reference and ari < 1.0 - 1e-12]
        initial_errors = sum(initial != reference for _, initial, reference in initial_comparable)
        final_errors = sum(predicted != reference for predicted, reference in final_comparable)
        output.append({
            "dataset": dataset,
            "scope": "full_pipeline" + (f"/{overlap_condition}" if overlap_condition else ""),
            "configuration": configuration,
            "recording_count": len(method_rows),
            "initial_count_comparable_recordings": len(initial_comparable),
            "initial_count_error_count": initial_errors if initial_comparable else "unavailable",
            "final_count_comparable_recordings": len(final_comparable),
            "final_count_error_count": final_errors,
            "initial_correct_count_wrong_partition_count": (
                len(wrong_partition) if partition_comparable else "unavailable"),
            "correct_count_with_speaker_error_count": "unavailable",
            "partition_evidence": ("initial_clustering_ARI" if partition_comparable
                                   else "unavailable_without_saved_initial_labels_or_count"),
            "robust_vs_fixed_regression_count": "not_applicable",
            "robust_vs_fixed_mean_recording_der_delta_pp": "not_applicable",
            "hierarchy_failure_type": "unavailable_without_integrated_merge_artifacts",
            "cross_corpus_der_comparable": False,
            "mechanism_evidence": ("correct_initial_count_wrong_partition_observed"
                                   if wrong_partition else
                                   "count_error_observed" if initial_errors or final_errors else
                                   "no_failure_in_available_metrics"),
        })

    fixed_name, robust_name = "fixed_ahc_075", "robust_adaptive_ahc_075"
    fixed = {row["recording_id"]: row for row in grouped.get(fixed_name, [])}
    robust = {row["recording_id"]: row for row in grouped.get(robust_name, [])}
    common = sorted(fixed.keys() & robust.keys())
    if common:
        deltas = [float(robust[key]["DER"]) - float(fixed[key]["DER"]) for key in common]
        robust_summary = next(row for row in output if row["configuration"] == robust_name)
        robust_summary["robust_vs_fixed_regression_count"] = sum(delta > 0 for delta in deltas)
        robust_summary["robust_vs_fixed_mean_recording_der_delta_pp"] = sum(deltas) / len(deltas)
    return output, grouped


def _summarize_viyt(path):
    rows = _read(path)
    if not rows:
        raise ValueError(f"{path}: no stage-count rows")
    initial_errors = sum(int(row["initial_K"]) != int(row["reference_K"]) for row in rows)
    final_errors = sum(int(row["final_K"]) != int(row["reference_K"]) for row in rows)
    correct_with_error = sum(
        int(row["final_K"]) == int(row["reference_K"]) and float(row["SpeakerError"]) > 0
        for row in rows)
    return [{
        "dataset": "ViYT-Diar",
        "scope": "saved_stage_count_analysis",
        "configuration": "artifact_configuration_unavailable",
        "recording_count": len(rows),
        "initial_count_comparable_recordings": len(rows),
        "initial_count_error_count": initial_errors,
        "final_count_comparable_recordings": len(rows),
        "final_count_error_count": final_errors,
        "initial_correct_count_wrong_partition_count": "unavailable",
        "correct_count_with_speaker_error_count": correct_with_error,
        "partition_evidence": "unavailable; SpeakerError is only a temporal assignment-error proxy",
        "robust_vs_fixed_regression_count": "unavailable",
        "robust_vs_fixed_mean_recording_der_delta_pp": "unavailable",
        "hierarchy_failure_type": "unavailable_without_integrated_merge_artifacts",
        "cross_corpus_der_comparable": False,
        "mechanism_evidence": "correct_final_count_with_speaker_error_observed",
    }]


def run(viyt_path=DEFAULT_VIYT, synthetic_path=DEFAULT_SYNTHETIC,
        natural_path=DEFAULT_NATURAL, output_dir=DEFAULT_OUTPUT):
    viyt = _summarize_viyt(viyt_path)
    synthetic, _ = _summarize_benchmark(
        "Synthetic Vietnamese holdout", synthetic_path)
    natural, natural_rows = _summarize_benchmark(
        "Natural Vietnamese DataHive", natural_path, "included")
    rows = viyt + synthetic + natural

    robust_vi4 = next((row for row in natural_rows.get("robust_adaptive_ahc_075", [])
                       if row["recording_id"] == "vi_000004"), None)
    vi4 = None if robust_vi4 is None else {
        "recording_id": "vi_000004",
        "reference_k": _reference_k(robust_vi4),
        "robust_initial_k": _initial_k(robust_vi4),
        "robust_final_k": int(robust_vi4["predicted_speakers"]),
        "initial_partition_ari": _number(robust_vi4.get("ARI")),
        "stage_warning": "initial ARI and final postprocessed K describe different stages",
    }
    summary = {
        "schema_version": 1,
        "sources": {
            "viyt_stage_counts": str(Path(viyt_path).resolve()),
            "synthetic_holdout_recordings": str(Path(synthetic_path).resolve()),
            "natural_datahive_recordings": str(Path(natural_path).resolve()),
        },
        "comparison_contract": {
            "labels_used_for_tuning": False,
            "cross_corpus_der_comparable": False,
            "initial_count_and_partition_comparison": "AHC estimated_n_speakers vs initial-clustering ARI",
            "final_count_comparison": "postprocessed predicted_speakers vs reference count",
            "robust_vs_fixed_deltas": "paired within each corpus only",
            "missing_fields": "explicitly marked unavailable or not_applicable",
        },
        "row_count": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "vi_000004": vi4,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "mechanisms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viyt-stage-counts", default=str(DEFAULT_VIYT))
    parser.add_argument("--synthetic-recordings", default=str(DEFAULT_SYNTHETIC))
    parser.add_argument("--natural-recordings", default=str(DEFAULT_NATURAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    rows, summary = run(args.viyt_stage_counts, args.synthetic_recordings,
                        args.natural_recordings, args.output_dir)
    print(json.dumps({"rows": len(rows), "datasets": summary["datasets"],
                      "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
