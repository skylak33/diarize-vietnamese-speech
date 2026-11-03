# ECAPA embedding and diarization error analysis

## Checkpoint compatibility correction

The 19-recording diagnostic run used an encoder whose Res2Net blocks applied
`Conv -> BatchNorm -> ReLU`. Checkpoint training and historical source commit
`a78321de6c46e86161099e99f3170ea2032d4403` use
`Conv -> ReLU -> BatchNorm`. Strict state-dictionary loading did not detect the
semantic mismatch because the same layers, names, and tensor shapes exist in
both graphs.

The saved 1,317 embeddings have a normalized centroid norm of `0.998002` and
were reproduced exactly by the wrong-order audit variant. They are valid
evidence about that faulty inference graph, but are invalid for assessing the
corrected encoder. Their one-speaker RTTMs, DER, SpeakerError, similarity
statistics, and AHC traces must not be presented as corrected-model results.
Regenerate embeddings before any new clustering or evaluation. The compatibility
fix itself is not a measured quality improvement.

The audit also corrected the distribution-overlap estimator. Across the same 19
wrong-order artifacts, fixed-range overlap was `1.0` for every recording, while
adaptive overlap ranged from `0.553817` to `0.914706` (median `0.837309`). The
diagnostic AUC calculation was unchanged (range `0.470921`-`0.734056`, median
`0.583723`). These pairwise statistics are descriptive because overlapping
windows and repeated pairs are dependent. Full methodology and reproduction
commands are in [ECAPA_CHECKPOINT_AUDIT.md](ECAPA_CHECKPOINT_AUDIT.md).

## Diagnostic architecture

With diagnostics enabled, inference writes
`<diagnostic_output_dir>/<recording_id>/` containing:

- `embeddings.npz`: CPU NumPy embeddings and window start/end times;
- `pipeline_trace.json`: labels after AHC, centroid merge, smoothing,
  partitioning, short-segment merging, and final RTTM;
- `config.json`: input hashes, effective configuration, protocol, reference
  source, and seed;
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

Run from the repository root. Replace the input WAV/reference and output
locations; the commands contain no machine-specific absolute paths.

```powershell
$audio = "data\recordings\98EV6H0ZRlU.wav"
$model = "exps\exp1\model\model_0034.model"
$referenceRttm = "data\references"
$artifacts = "results\embedding_analysis"

python -m diarize.pipeline $audio `
  --model $model `
  --method ahc --threshold 0.75 `
  --output "$artifacts\98EV6H0ZRlU.rttm" `
  --diagnostic_output_dir $artifacts `
  --diagnostic_counterfactuals `
  --evaluation_protocol "ViYT-Diar collar=0.25 include-overlap" `
  --reference_source published_reference

python -m experiments.analyze_embedding_diagnostics `
  --artifacts $artifacts `
  --reference_rttm $referenceRttm `
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

## Pre-correction recording ledger

The following labels describe the original investigation plan and wrong-order
run. They are retained only to identify the recordings; none is a corrected-model
result. Regenerate the complete corpus before updating speaker counts or DER.

| Recording | Group | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| `98EV6H0ZRlU` | historical comparison | invalidated | wrong-order embeddings | regenerate |
| `B5rLHPUJ8qc` | historical comparison | invalidated | wrong-order embeddings | regenerate |
| `jxTaydBnwIg` | historical comparison | invalidated | wrong-order embeddings | regenerate |
| `XSC94UbpoRQ` | historical comparison | invalidated | wrong-order embeddings | regenerate |
| `Hx0j-Oc18do` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `9lm02QThHjo` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `sVPA0POzHFQ` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `IZxOT44FKsU` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `HTwksOmzTsc` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `kmP3mlKeGRY` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `_5qHkZ0miYY` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `XJG-o9BK4To` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `lDvnabH6iyY` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `kkPMThxTwq0` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `hRYWiO-zi5s` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `kBbOVOaU6cc` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `LNdIO_4fRuI` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `sxOvp1cgE14` | wrong-order undercount | invalidated | wrong-order embeddings | regenerate |
| `jE9ZB-zqEgs` | historical comparison | invalidated | wrong-order embeddings | regenerate |

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
