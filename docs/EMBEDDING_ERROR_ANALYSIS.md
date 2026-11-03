# ECAPA embedding and diarization error analysis

## Implementation status versus experimental findings

The diagnostic implementation is present in the current commits. It can write
per-recording artifacts, trace AHC and temporal stages, compute similarity
statistics, and render four plots. The checkpoint
`exps/exp1/model/model_0034.model` is present in this checkout. The 19 target
recordings listed below are not available as local WAV inputs, so no real-data
embedding, AHC, or postprocessing findings are claimed here. The WAV files
under third-party package test directories are unrelated fixtures.

The current test suite exercises synthetic inputs only. Synthetic diagnostic
tests are implemented and can be run now; target-recording analysis is pending
until the corresponding WAV/reference inputs are supplied. A successful test
run therefore demonstrates implementation behavior, not speaker separation on
ViYT-Diar.

## Diagnostic architecture

With diagnostics enabled, inference writes
`<diagnostic_output_dir>/<recording_id>/` containing:

- `embeddings.npz`: CPU NumPy embeddings and window start/end times;
- `pipeline_trace.json`: configuration, hashes, labels after AHC, centroid
  merge, smoothing, partitioning, short-segment merging, and final RTTM;
- `config.json`: reproducibility metadata;
- `ahc_merges.json`: the recorded hierarchy, children, distances, merge steps,
  and cluster sizes.

The diagnostic path preserves the production RTTM. Counterfactual outputs are
written under `counterfactuals/<recording_id>/` and keep audio, VAD, embeddings,
and AHC fixed while disabling one temporal operation. They are attribution
experiments, not production settings.

AHC distances are cosine distances. The merge trace records
`merge_step`, `left_cluster_id`, `right_cluster_id`, `merge_distance`, and
`merged_cluster_size`; it is the actual scikit-learn hierarchy, not an
approximation. Cluster IDs are arbitrary, so comparisons must use partitions or
reference-independent mappings rather than numeric label equality.

Similarity statistics distinguish within/between *predicted clusters* from
same/different *reference speakers*. Reference-aware windows use temporal
coverage, not only a midpoint: windows can be `unambiguous_speaker_window`,
`mixed_speaker_window`, `overlapping_speech_window`, `unannotated_window`, or
`uncertain_window`. High cosine similarity is not proof of identical speakers,
and low similarity is not proof of different speakers.

The visualizer produces four independent views:

1. chronological cosine-similarity heatmap;
2. reference and pipeline-stage speaker timeline;
3. predicted-cluster and optional reference-speaker similarity distributions;
4. the exact AHC dendrogram with its threshold.

## Windows PowerShell workflow

Run from the repository root. Replace only the input WAV and output locations;
the commands contain no machine-specific absolute paths.

```powershell
$audio = "data\recordings\98EV6H0ZRlU.wav"
$model = "exps\exp1\model\model_0034.model"
$artifacts = "results\embedding_analysis"

python -m diarize.pipeline $audio `
  --model $model `
  --method ahc --threshold 0.75 `
  --output "$artifacts\98EV6H0ZRlU.rttm" `
  --diagnostic_output_dir $artifacts `
  --diagnostic_counterfactuals

python -m experiments.analyze_embedding_diagnostics `
  --artifacts $artifacts `
  --recording_id 98EV6H0ZRlU `
  --output_dir $artifacts

python -m experiments.visualize_embedding_diagnostics `
  "$artifacts\98EV6H0ZRlU" `
  --output_dir "$artifacts\98EV6H0ZRlU" `
  --format png
```

Use `--format svg` for vector figures. Analysis and visualization reuse saved
artifacts; they do not invoke ECAPA again. For a corpus, run the first command
once per recording and pass the common artifact root to the analysis command.

## Reproducibility and cache

Results are interpretable only with the audio SHA-256, checkpoint SHA-256,
seed, effective inference configuration, evaluation protocol, and reference
source recorded beside the artifact. The embedding cache may reuse compatible
audio/checkpoint/VAD/window/device/seed entries; changing segmentation or VAD
must create a new cache key. Save generated arrays and figures outside Git.

The existing ViYT-Diar test set has already been used for exploratory tuning.
Do not select AHC, VAD, window, smoothing, or merge parameters from these
diagnostics, and do not report counterfactual DER as a new production result.
Use a speaker-disjoint development set and an untouched held-out test set for
future selection.

## Target recording ledger

All rows below are intentionally marked `not run / WAV unavailable`. Evidence is
`unknown`; no cause is inferred from the speaker-count group. `N_ref = 2` and
`N_pred = 1` are historical group labels for the 14 undercount cases, not new
measurements in this checkout.

| Recording | Group | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| `98EV6H0ZRlU` | correct count | not run / WAV unavailable | unknown | pending WAV/reference |
| `B5rLHPUJ8qc` | correct count | not run / WAV unavailable | unknown | pending WAV/reference |
| `jxTaydBnwIg` | correct count | not run / WAV unavailable | unknown | pending WAV/reference |
| `XSC94UbpoRQ` | correct count | not run / WAV unavailable | unknown | pending WAV/reference |
| `Hx0j-Oc18do` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `9lm02QThHjo` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `sVPA0POzHFQ` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `IZxOT44FKsU` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `HTwksOmzTsc` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `kmP3mlKeGRY` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `_5qHkZ0miYY` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `XJG-o9BK4To` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `lDvnabH6iyY` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `kkPMThxTwq0` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `hRYWiO-zi5s` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `kBbOVOaU6cc` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `LNdIO_4fRuI` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `sxOvp1cgE14` | undercount | not run / WAV unavailable | unknown | historical N_ref=2, N_pred=1 |
| `jE9ZB-zqEgs` | overcount | not run / WAV unavailable | unknown | additional case; pending WAV/reference |

Manual references may contain broad turns with pauses and are not guaranteed
frame-perfect speech activity. Published and manually audited references must
remain separate. Approximate listening notes cannot be promoted to RTTM or UEM
ground truth. Reference overlap or incomplete coverage can make
speaker-level evidence indeterminate.

## Validation boundary

Run `python -m pytest -q` and
`python -m compileall diarize eval experiments tests` for implementation
checks. These checks do not establish real-data findings. Once the target WAVs,
references, and protocol manifest are available, archive the generated JSON,
CSV, NPZ, PNG/SVG artifacts together with their hashes and report confirmed,
hypothesized, and unknown causes separately.
