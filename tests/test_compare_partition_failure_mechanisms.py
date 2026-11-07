import csv
import json

from experiments.compare_partition_failure_mechanisms import run


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_row(configuration, recording, reference, initial, final, ari, der,
                   overlap=None):
    row = {
        "mode": "full_pipeline", "configuration": configuration,
        "recording_id": recording, "reference_speaker_count": reference,
        "predicted_speakers": final, "ARI": ari, "DER": der,
        "clustering_diagnostics": json.dumps(
            {"ahc": {"estimated_n_speakers": initial}}),
    }
    if overlap is not None:
        row["overlap_condition"] = overlap
    return row


def test_saved_artifacts_are_compared_by_stage_without_cross_corpus_der_ranking(tmp_path):
    viyt = tmp_path / "viyt.csv"
    synthetic = tmp_path / "synthetic.csv"
    natural = tmp_path / "natural.csv"
    output = tmp_path / "output"
    _write_csv(viyt, [{
        "recording": "v", "reference_K": 2, "initial_K": 3, "final_K": 2,
        "SpeakerError": 4.0,
    }])
    _write_csv(synthetic, [
        _benchmark_row("fixed_ahc_075", "s", 2, 2, 2, 1.0, 5.0),
        _benchmark_row("robust_adaptive_ahc_075", "s", 2, 2, 2, 0.0, 8.0),
    ])
    _write_csv(natural, [
        _benchmark_row("fixed_ahc_075", "vi_000004", 2, 7, 7, 0.8, 15.0, "included"),
        _benchmark_row("robust_adaptive_ahc_075", "vi_000004", 2, 4, 2, 0.2, 46.0, "included"),
    ])

    rows, summary = run(viyt, synthetic, natural, output)
    synthetic_robust = next(row for row in rows
                            if row["dataset"] == "Synthetic Vietnamese holdout"
                            and row["configuration"] == "robust_adaptive_ahc_075")
    natural_robust = next(row for row in rows
                          if row["dataset"] == "Natural Vietnamese DataHive"
                          and row["configuration"] == "robust_adaptive_ahc_075")
    assert synthetic_robust["initial_correct_count_wrong_partition_count"] == 1
    assert synthetic_robust["robust_vs_fixed_regression_count"] == 1
    assert natural_robust["initial_count_error_count"] == 1
    assert natural_robust["final_count_error_count"] == 0
    assert all(row["cross_corpus_der_comparable"] is False for row in rows)
    assert summary["vi_000004"]["robust_initial_k"] == 4
    assert summary["vi_000004"]["robust_final_k"] == 2
    assert (output / "mechanisms.csv").is_file()
    assert json.loads((output / "summary.json").read_text())["comparison_contract"][
        "labels_used_for_tuning"] is False
