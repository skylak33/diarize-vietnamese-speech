"""Create grouped synthetic benchmark comparisons and robust-AHC regressions."""

import argparse
import csv
import json
from pathlib import Path


FIXED = "fixed_ahc_075"
ROBUST = "robust_adaptive_ahc_075"


def _mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _pooled(rows, seconds_key):
    duration = sum(row["scored_duration_seconds"] for row in rows)
    return 100 * sum(row[seconds_key] for row in rows) / duration if duration else None


def _partition_mean(rows, key):
    available = [row for row in rows if row.get(key) is not None]
    denominator = sum(row["compatible_windows"] for row in available)
    return (sum(row[key] * row["compatible_windows"] for row in available) / denominator
            if denominator else None)


def _bin(value, lower, upper):
    return f"<{lower:g}" if value < lower else (f"{lower:g}-{upper:g}" if value <= upper
                                                  else f">{upper:g}")


def grouped_comparisons(rows):
    """Aggregate by fixed metadata groups; bins are reporting-only, never tuning."""
    groups = {}
    for row in rows:
        dimensions = (("speaker_count", row["reference_speaker_count"]),
                      ("scenario", row.get("scenario")),
                      ("duration_seconds", _bin(row["duration_seconds"], 45, 75)),
                      ("mean_turn_duration_seconds",
                       _bin(row["mean_turn_duration_seconds"], 2.5, 5)))
        dimensions += tuple(("source_dataset", value) for value in row.get("source_datasets", []))
        for group_type, value in dimensions:
            key = (row["split"], row["mode"], row["configuration"], group_type, value)
            groups.setdefault(key, []).append(row)
    output = []
    for (split, mode, configuration, group_type, value), members in sorted(
            groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        components = [_pooled(members, key) for key in
                      ("miss_seconds", "false_alarm_seconds", "speaker_error_seconds")]
        output.append({
            "split": split, "mode": mode, "configuration": configuration,
            "group_type": group_type, "group_value": value, "recordings": len(members),
            "DER": sum(components) if all(value is not None for value in components) else None,
            "Miss": components[0], "FalseAlarm": components[1], "SpeakerError": components[2],
            "speaker_count_accuracy": _mean(members, "count_correct"),
            "speaker_count_MAE": _mean(members, "count_absolute_error"),
            "overclustering_rate": _mean(members, "overclustered"),
            "underclustering_rate": _mean(members, "underclustered"),
            "runtime_seconds": sum(row["runtime_seconds"] for row in members),
            "ARI": _partition_mean(members, "ARI"), "NMI": _partition_mean(members, "NMI"),
            "purity": _partition_mean(members, "purity"),
            "fragmentation": _partition_mean(members, "fragmentation"),
        })
    return output


def robust_vs_fixed(rows):
    """Return positive DER regressions and a summary of all paired deltas."""
    indexed = {(row["split"], row["mode"], row["recording_id"], row["configuration"]): row
               for row in rows if row["configuration"] in {FIXED, ROBUST}}
    pairs = []
    identities = sorted({key[:3] for key in indexed})
    for identity in identities:
        fixed, robust = indexed.get(identity + (FIXED,)), indexed.get(identity + (ROBUST,))
        if fixed is None or robust is None or fixed["DER"] is None or robust["DER"] is None:
            continue
        delta = robust["DER"] - fixed["DER"]
        pairs.append({"split": identity[0], "mode": identity[1], "recording_id": identity[2],
                      "scenario": robust.get("scenario"),
                      "reference_speaker_count": robust["reference_speaker_count"],
                      "source_datasets": robust.get("source_datasets", []),
                      "fixed_DER": fixed["DER"], "robust_DER": robust["DER"],
                      "DER_delta_robust_minus_fixed": delta,
                      "count_absolute_error_delta": (robust["count_absolute_error"]
                                                     - fixed["count_absolute_error"])})
    regressions = sorted((row for row in pairs if row["DER_delta_robust_minus_fixed"] > 1e-12),
                         key=lambda row: (-row["DER_delta_robust_minus_fixed"], row["recording_id"]))

    def counts(items):
        deltas = [row["DER_delta_robust_minus_fixed"] for row in items]
        return {"paired_recordings": len(items),
                "improved": sum(delta < -1e-12 for delta in deltas),
                "worsened": sum(delta > 1e-12 for delta in deltas),
                "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas),
                "mean_DER_delta_robust_minus_fixed": sum(deltas) / len(deltas) if deltas else None}

    summary = counts(pairs)
    summary["by_mode"] = {mode: counts([row for row in pairs if row["mode"] == mode])
                          for mode in sorted({row["mode"] for row in pairs})}
    summary["by_reference_speaker_count"] = {
        str(speaker_count): counts([row for row in pairs
                                    if row["reference_speaker_count"] == speaker_count])
        for speaker_count in sorted({row["reference_speaker_count"] for row in pairs})}
    summary["worst_cases"] = regressions[:10]
    return regressions, summary


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def analyze(results_path, output_dir=None):
    results_path = Path(results_path)
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    splits = {row["split"] for row in rows}
    if len(splits) != 1:
        raise ValueError(f"Expected one split, found {sorted(splits)}")
    output = Path(output_dir) if output_dir else results_path.parent
    output.mkdir(parents=True, exist_ok=True)
    grouped = grouped_comparisons(rows)
    regressions, summary = robust_vs_fixed(rows)
    _write_csv(output / "grouped_comparisons.csv", grouped)
    _write_csv(output / "robust_vs_fixed_regressions.csv", regressions)
    (output / "robust_vs_fixed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return grouped, regressions, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_results")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    analyze(args.recording_results, args.output_dir)


if __name__ == "__main__":
    main()
