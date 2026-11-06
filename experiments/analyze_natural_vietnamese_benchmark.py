"""Analyze the frozen natural Vietnamese diarization benchmark."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("fixed_ahc_075", "robust_adaptive_ahc_075"),
    ("fixed_ahc_065", "fixed_ahc_075"),
)


def _pooled_der(rows):
    duration = sum(row["scored_duration_seconds"] for row in rows)
    errors = sum(row["miss_seconds"] + row["false_alarm_seconds"]
                 + row["speaker_error_seconds"] for row in rows)
    return 100 * errors / duration


def paired_bootstrap(rows, baseline, candidate, draws=10_000, seed=42):
    """Return candidate-minus-baseline pooled-DER deltas by protocol stratum."""
    if draws < 1:
        raise ValueError("draws must be positive")
    indexed = {(row["mode"], row["overlap_condition"], row["recording_id"],
                row["configuration"]): row for row in rows}
    strata = sorted({key[:2] for key in indexed})
    output = []
    rng = np.random.default_rng(seed)
    for mode, overlap in strata:
        recording_ids = sorted({key[2] for key in indexed
                                if key[:2] == (mode, overlap)})
        pairs = [(indexed.get((mode, overlap, recording_id, baseline)),
                  indexed.get((mode, overlap, recording_id, candidate)))
                 for recording_id in recording_ids]
        if any(left is None or right is None for left, right in pairs):
            raise ValueError(f"Incomplete pairing for {mode}/{overlap}")
        baseline_rows = [pair[0] for pair in pairs]
        candidate_rows = [pair[1] for pair in pairs]
        deltas = np.asarray([right["DER"] - left["DER"]
                             for left, right in pairs], dtype=float)
        samples = rng.integers(0, len(pairs), size=(draws, len(pairs)))
        bootstrap = np.empty(draws, dtype=float)
        for index, sample in enumerate(samples):
            bootstrap[index] = (_pooled_der([candidate_rows[i] for i in sample])
                                - _pooled_der([baseline_rows[i] for i in sample]))
        output.append({
            "mode": mode,
            "overlap_condition": overlap,
            "baseline": baseline,
            "candidate": candidate,
            "recordings": len(pairs),
            "baseline_DER": _pooled_der(baseline_rows),
            "candidate_DER": _pooled_der(candidate_rows),
            "DER_delta_candidate_minus_baseline": (
                _pooled_der(candidate_rows) - _pooled_der(baseline_rows)),
            "bootstrap_ci95_low": float(np.percentile(bootstrap, 2.5)),
            "bootstrap_ci95_high": float(np.percentile(bootstrap, 97.5)),
            "improved": int(np.sum(deltas < -1e-12)),
            "worsened": int(np.sum(deltas > 1e-12)),
            "unchanged": int(np.sum(np.abs(deltas) <= 1e-12)),
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
        })
    return output


def count_confusion(rows):
    counts = {}
    for row in rows:
        key = (row["mode"], row["overlap_condition"], row["configuration"],
               row["reference_speaker_count"], row["predicted_speakers"])
        counts[key] = counts.get(key, 0) + 1
    return [{"mode": key[0], "overlap_condition": key[1],
             "configuration": key[2], "reference_speakers": key[3],
             "predicted_speakers": key[4], "recordings": value}
            for key, value in sorted(counts.items())]


def failure_cases(rows, top_per_stratum=3):
    selected = {}
    groups = {}
    for row in rows:
        key = (row["mode"], row["overlap_condition"], row["configuration"])
        groups.setdefault(key, []).append(row)
    for members in groups.values():
        top_ids = {row["recording_id"] for row in
                   sorted(members, key=lambda item: -item["DER"])[:top_per_stratum]}
        for row in members:
            reasons = []
            if not row["count_correct"]:
                reasons.append("speaker_count_mismatch")
            if row["recording_id"] in top_ids:
                reasons.append("top3_DER_in_stratum")
            if reasons:
                key = (row["mode"], row["overlap_condition"], row["configuration"],
                       row["recording_id"])
                selected[key] = {
                    "mode": row["mode"],
                    "overlap_condition": row["overlap_condition"],
                    "configuration": row["configuration"],
                    "recording_id": row["recording_id"],
                    "reference_speakers": row["reference_speaker_count"],
                    "predicted_speakers": row["predicted_speakers"],
                    "DER": row["DER"],
                    "Miss": row["Miss"],
                    "FalseAlarm": row["FalseAlarm"],
                    "SpeakerError": row["SpeakerError"],
                    "reasons": ";".join(reasons),
                }
    return [selected[key] for key in sorted(selected)]


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    included = [row for row in rows if row["overlap_condition"] == "included"]
    configurations = sorted({row["configuration"] for row in included})
    labels = [name.replace("_adaptive_ahc_075", " adaptive").replace("_ahc_", " ")
              for name in configurations]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, mode in zip(axes, ("full_pipeline", "clustering_only")):
        axis.boxplot([[row["DER"] for row in included
                       if row["mode"] == mode and row["configuration"] == config]
                      for config in configurations], tick_labels=labels)
        axis.set_title(mode.replace("_", " "))
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Per-recording DER (%)")
    fig.tight_layout()
    fig.savefig(output / "der_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for axis, mode in zip(axes, ("full_pipeline", "clustering_only")):
        fixed = {row["recording_id"]: row for row in included
                 if row["mode"] == mode and row["configuration"] == "fixed_ahc_075"}
        robust = {row["recording_id"]: row for row in included
                  if row["mode"] == mode
                  and row["configuration"] == "robust_adaptive_ahc_075"}
        ids = sorted(set(fixed) & set(robust))
        x = [fixed[item]["DER"] for item in ids]
        y = [robust[item]["DER"] for item in ids]
        limit = max(x + y) + 2
        axis.scatter(x, y)
        axis.plot([0, limit], [0, limit], "--", color="grey")
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("Fixed 0.75 DER (%)")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Robust adaptive DER (%)")
    fig.tight_layout()
    fig.savefig(output / "fixed_vs_robust_paired.png", dpi=150)
    plt.close(fig)


def analyze(results_path, output_dir=None, draws=10_000, seed=42):
    results_path = Path(results_path)
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    output = Path(output_dir) if output_dir else results_path.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for baseline, candidate in COMPARISONS:
        comparisons.extend(paired_bootstrap(rows, baseline, candidate, draws, seed))
    confusion = count_confusion(rows)
    failures = failure_cases(rows)
    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8")
    _write_csv(output / "speaker_count_confusion.csv", confusion)
    _write_csv(output / "failure_cases.csv", failures)
    _plots(rows, output)
    return comparisons, confusion, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_results")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    analyze(args.recording_results, args.output_dir, args.bootstrap_draws, args.seed)


if __name__ == "__main__":
    main()
