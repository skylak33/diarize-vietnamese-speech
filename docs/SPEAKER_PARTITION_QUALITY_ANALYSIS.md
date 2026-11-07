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

## Verified saved-artifact evidence

The completed DataHive manifest contains eight recordings at dataset revision `c29728a0c9792ba486807fa1ea9ca64e292cec01`. It records checkpoint SHA256 `58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`, configuration SHA256 `0e006d5c23eed690d5ede5590b872460a3d2725f44a3809f002f07f1887addbb`, annotation-manifest SHA256 `723349f5dbe5993af655871417f9d881d0f64371c815764102dec5752a2ed03c`, both `full_pipeline` and `clustering_only` modes, and included/excluded-overlap scoring. The reference provenance is `silver_reference`.

Saved stage-count evidence must be read by stage. On `vi_000004`, Robust AHC selected **initial K=4**, not K=2. Temporal postprocessing reduced the output to **final K=2**. The saved initial-clustering ARI is `0.0009618646`; it must not be described as the quality of the final two-speaker partition without a stage-aligned recomputation.

The stability artifacts contain 16 recording/mode groups per method and perturbation family. Averaging the saved group summaries, Robust angular-noise runs changed K in `0.5069` of runs and had mean ARI `0.6678` against the unperturbed partition. Robust short-prefix runs changed K in `0.8438` of runs, with mean ARI `0.1440` and mean changed-assignment fraction `0.4288`. Fixed 0.65 had no angular-noise count changes and mean ARI `0.9004`, but its short-prefix count-change fraction was `0.9792`. These are stability measurements, not correctness rankings. The controlled `correct_k_wrong_partition` case gives K=2 with ARI `-0.5` for Fixed, Legacy, and Robust, directly demonstrating that correct count does not guarantee a correct partition.

The saved silver-reference audit covers all eight DataHive recordings. Its fixed rules produced 432 simultaneous-activity intervals totaling 347.3 seconds, 438 suspicious intervals, and 1,586 high-confidence channel-separated windows. The review manifest contains 24 items and explicitly records `human_verified: false`; no suspicious interval is treated as a corrected label or proof of leakage.

The cross-dataset comparison reads saved CSV/JSON only. It marks raw DER as non-comparable across corpora, compares Robust-versus-Fixed deltas only within a corpus, and leaves hierarchy failure types unavailable until integrated merge artifacts exist. ViYT-Diar supplies stage counts and SpeakerError only, so its partition evidence remains unavailable rather than inferred from a temporal-error proxy.

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

Reproduce the saved-cache stability and cross-dataset summaries without audio inference:

```powershell
python experiments/analyze_ahc_merge_quality.py
python experiments/analyze_clustering_stability.py
python experiments/compare_partition_failure_mechanisms.py
python -m pytest -q tests/test_ahc_merge_diagnostics.py tests/test_clustering_stability.py tests/test_compare_partition_failure_mechanisms.py
```

The integrated analysis reads the verified embedding-cache manifest and saved benchmark traces. It writes 96 per-recording rows, 59,836 merge rows, 2,012 reference-crossing merge rows, and six representative figures without rerunning ECAPA.

## Observation ledger

Complete one row per material observation. Never promote a hypothesis to a confirmed finding without direct computational or manual evidence.

| ID | Dataset / recording | Method / configuration | Direct observation | Reference-dependent measurement | Hypothesized cause | Evidence artifact | Confidence | Status |
|---|---|---|---|---|---|---|---|---|
| OBS-001 | `vi_000004` | Fixed 0.75 vs Robust 0.75 | Robust initial K=4; final K=2; Fixed initial/final K=10/7 | Robust initial ARI=0.001 vs Fixed 0.827; oracle K=5 gives 0.934 | Early mixing plus catastrophic K=5-to-4 selected-cut merge | `partition_quality/summary.json` | High for computation; silver-reference limited | Verified |
| OBS-002 | `vi_000007` | Fixed 0.75 vs Robust 0.75 | Robust K=2 to K=1 after postprocessing | Robust initial ARI about 0; oracle K=7 gives 0.794 | Early mixing, poor cut, identity elimination | `partition_quality/summary.json` | High for computation; silver-reference limited | Verified |
| OBS-003 | `vi_000001` | Fixed 0.75 vs Robust 0.75 | Robust K=2 to K=1 after postprocessing | Robust initial ARI=0.002; oracle K=4 gives 0.932 | Early mixing, poor cut, identity elimination | `partition_quality/summary.json` | High for computation; silver-reference limited | Verified |
| OBS-004 | `vi_000005` | Fixed 0.75 vs Robust 0.75 | Robust K=2 to K=1 after postprocessing | Robust initial ARI=-0.005; oracle K=6 gives 0.909 | Early mixing, poor cut, identity elimination | `partition_quality/summary.json` | High for computation; silver-reference limited | Verified |
| OBS-005 | `vi_000008` | Fixed 0.75 vs Robust 0.75 | Robust K=2 to K=1 after postprocessing | Robust initial ARI=-0.002; oracle K=3 gives 0.961 | Early mixing, poor cut, identity elimination | `partition_quality/summary.json` | High for computation; silver-reference limited | Verified |

Each final classification may contain multiple mechanisms: incorrect hierarchy cut, incorrect early merge, speaker fragmentation, embedding ambiguity, temporal postprocessing, reference uncertainty, overlapping-speech limitation, or unresolved.

## Pending results

- **Baseline provenance:** saved DataHive manifest verified; independent rerun remains pending.
- **Partition-quality comparison:** complete for saved DataHive initial AHC labels; Spectral/NME-SC retain aggregate-only initial metrics.
- **Cross-speaker merge frequency:** complete as an oracle diagnostic over K=1...10 cuts.
- **Perturbation and short-hierarchy stability:** saved artifacts summarized above; correctness interpretation remains pending integrated partition/hierarchy evidence.
- **DataHive manual listening:** pending; no human-verified labels are claimed.
- **Cross-dataset interpretation and recommendation:** complete at mechanism level; raw DER remains non-comparable across corpora.
