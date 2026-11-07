# Speaker Partition Quality and Hierarchical Clustering Stability

## Status

This document defines the frozen research protocol and the reporting ledger. Real-data conclusions are pending completion of the analysis. DataHive annotations are silver references derived from independent channel-wise VAD; they are not manually verified ground truth.

## Research question

The investigation asks why an adaptive AHC estimator can return the correct number of speakers while returning the wrong speaker partition. Speaker-count accuracy is therefore reported separately from assignment quality, purity, fragmentation, hierarchy quality, cut stability, temporal postprocessing effects, and reference uncertainty.

No production estimator is developed or tuned in this study. The ECAPA checkpoint, VAD defaults, embedding windows, clustering defaults, smoothing, and historical benchmark configurations remain frozen.

## Data and reference contract

- The encoder is `exps/exp1/model/model_0034.model`, SHA256 `58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.
- The ECAPA `Bottle2neck` order is `conv1 -> ReLU -> bn1`.
- Compatible cached embeddings must be reused. Audio, reference, checkpoint, encoder-source, configuration, and Git hashes belong in the provenance manifest.
- Official DER uses the complete scoring regions and the frozen collar/overlap protocol. Ambiguous or overlapping windows are never removed from official DER.
- Reference-dependent window diagnostics use timestamp intersection. A window overlapping more than one reference speaker is marked ambiguous and excluded from single-speaker partition metrics; it is not assigned to the largest overlap for convenience.
- DataHive references remain explicitly labeled `silver_reference`. Signal correlation, energy leakage, and simultaneous channel activity are indicators of uncertainty, not corrected labels or proof of a physical cause.
- Oracle reference labels may explain a hierarchy after inference, but must never influence automatic clustering, speaker-count estimation, or parameter selection.

## Three distinct questions

1. **Count:** Does the method return the reference number of speakers? Report exact count accuracy and absolute count error.
2. **Partition:** Do the same windows belong together? Report ARI, NMI, duration-weighted purity, entropy, speaker confusion, and fragmentation. Correct count does not imply a correct partition.
3. **Hierarchy and cut:** Did AHC make an irreversible cross-speaker merge before the selected cut, or did a usable reference partition exist in the hierarchy but the estimator choose the wrong cut?

An **incorrect cut** means that another cut in the same hierarchy exposes a useful speaker partition. An **incorrect early merge** means distinct speakers were already mixed below that cut; no later AHC cut can undo the merge. A supervised cut sweep is an oracle diagnostic of representational capacity only, never a production selection rule.

## Frozen analysis protocol

1. Validate provenance and reproduce saved summaries without rerunning GPU inference when compatible artifacts exist.
2. Compute initial and postprocessed speaker counts, partition metrics, temporal errors, and embedding geometry from identical windows for every compared method.
3. Trace every AHC merge, including distance, sizes, reference composition, entropy, purity change, and whether it crosses reference speakers.
4. Separate incorrect-cut and incorrect-early-merge cases. Analyze `vi_000004` and the five largest Robust-versus-Fixed regressions selected mechanically from the results.
5. Measure angular-noise, ordered-window-subsampling, short-recording, count, cut, and partition stability with fixed seeds and predefined perturbation magnitudes.
6. Run controlled synthetic cases for clear separation, fragmentation, similar speakers, outliers, unequal sizes, short hierarchies, and correct-count/wrong-partition behavior.
7. Audit stereo alignment, simultaneous activity, lagged correlation, energy ratios, and suspicious intervals. Keep manual listening status explicit.
8. Compare failure mechanisms, not raw DER magnitudes, across ViYT-Diar, the synthetic holdout, and natural DataHive.

Stability and correctness are independent: a stable partition can be consistently wrong. Report perturbation agreement separately from reference-dependent quality.

## No-tuning restrictions

Do not retrain ECAPA, alter the checkpoint, VAD, embedding windows, production clustering defaults, or smoothing. Do not select thresholds, perturbation magnitudes, gap-margin rules, or high-confidence subsets based on which method wins on ViYT-Diar, DataHive, or the synthetic holdout. Do not download new datasets or silently rewrite references. Eight two-speaker DataHive conversations cannot establish generalization or support strong significance claims.

## Artifacts and reproduction

Expected outputs live under `results/partition_quality_stability/` and must not overwrite historical results:

```text
provenance/
partition_quality/{per_recording.csv,per_cluster.csv,per_speaker.csv,summary.json}
hierarchy/{merge_analysis.csv,cross_speaker_merges.csv,selected_cuts.json}
stability/{perturbation_results.csv,count_stability.csv,partition_stability.csv}
silver_reference_audit/{channel_diagnostics.csv,suspicious_intervals.csv,human_review_manifest.json}
figures/
report.md
```

Run lightweight validation from the repository root:

```powershell
python -m pytest -q tests/test_partition_analysis_invariants.py tests/test_partition_quality.py tests/test_ahc_merge_diagnostics.py tests/test_clustering_stability.py tests/test_silver_reference_audit.py
python -m compileall diarize eval experiments tests
git diff --check
```

Inspect the frozen benchmark plan without inference:

```powershell
python experiments/run_natural_vietnamese_benchmark.py --dry-run --config configs/natural_vietnamese_benchmark.json
```

Run the silver-reference audit using existing local annotations and audio. Audio excerpts remain opt-in and local:

```powershell
python experiments/audit_silver_reference.py
python experiments/audit_silver_reference.py --write-excerpts
```

The exact partition, hierarchy, stability, and comparison commands must be copied into `results/partition_quality_stability/provenance/` when their integrated CLIs are finalized. **Pending:** final CLI paths and effective arguments.

## Observation ledger

Complete one row per material observation. Never promote a hypothesis to a confirmed finding without direct computational or manual evidence.

| ID | Dataset / recording | Method / configuration | Direct observation | Reference-dependent measurement | Hypothesized cause | Evidence artifact | Confidence | Status |
|---|---|---|---|---|---|---|---|---|
| OBS-001 | `vi_000004` | Fixed 0.75 vs Robust 0.75 | Pending | Pending | Pending | Pending | Pending | Pending |
| OBS-002 | Largest regression 2 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| OBS-003 | Largest regression 3 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| OBS-004 | Largest regression 4 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| OBS-005 | Largest regression 5 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Each final classification may contain multiple mechanisms: incorrect hierarchy cut, incorrect early merge, speaker fragmentation, embedding ambiguity, temporal postprocessing, reference uncertainty, overlapping-speech limitation, or unresolved.

## Pending results

- **Baseline reproduction:** pending provenance verification.
- **Partition-quality comparison:** pending real-data artifacts.
- **Cross-speaker merge frequency:** pending hierarchy traces.
- **Perturbation and short-hierarchy stability:** pending experiments.
- **DataHive manual listening:** pending; no human-verified labels are claimed.
- **Cross-dataset interpretation and recommendation:** pending integrated evidence.
