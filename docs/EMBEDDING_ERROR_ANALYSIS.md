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
They remain isolated under `results/embedding_analysis`. The corrected run
regenerated all 1,317 windows under `results/embedding_analysis_corrected` and
reproduced the complete historical RTTM set; the compatibility fix by itself is
still not a quality-improvement claim.

The audit also corrected the distribution-overlap estimator. Across the same 19
wrong-order artifacts, fixed-range overlap was `1.0` for every recording, while
adaptive overlap ranged from `0.553817` to `0.914706` (median `0.837309`). The
diagnostic AUC calculation was unchanged (range `0.470921`-`0.734056`, median
`0.583723`). These pairwise statistics are descriptive because overlapping
windows and repeated pairs are dependent. Full methodology and reproduction
commands are in [ECAPA_CHECKPOINT_AUDIT.md](ECAPA_CHECKPOINT_AUDIT.md).

## Corrected 19-recording diagnostics

The corrected artifacts use encoder-source SHA-256
`96cd2147b769f3730d11eaabefedb3f3b52c3913f87f74d01b208d4bf92b14d8`
and the unchanged checkpoint. Across the 19 per-recording reference-aware
summaries, the mean same-speaker cosine was 0.4973 and the mean
different-speaker cosine was 0.3075. Per-recording diagnostic AUC had mean
0.8555, median 0.8508, and range 0.6307-0.9834; median KS was 0.5905,
median Wasserstein distance 0.1801, and median distribution overlap 0.4131.
These unweighted summaries describe the selected recordings and are not a
corpus-wide or independent-sample performance estimate.

The corrected count relations are 4 correct-count, 14 undercount, and 1
overcount. Reference-aware merge tracing found at least one merge joining
reference-distinct speakers in every selected recording, but reference overlap,
window mixing, and label alignment limit causal attribution. The automated
summary assigns low confidence to 18 cases and unknown confidence to
`jxTaydBnwIg`; it confirms no physical root cause. AHC overmerging,
fragmentation, mixed windows, and smoothing are candidate mechanisms for manual
review, not confirmed explanations.

| Recording | Count relation | SpeakerError s | AUC | Reference-distinct merges | Initial/final clusters |
| --- | --- | ---: | ---: | ---: | ---: |
| `98EV6H0ZRlU` | correct | 15.312 | 0.940 | 1 | 2/2 |
| `B5rLHPUJ8qc` | correct | 11.810 | 0.825 | 1 | 3/2 |
| `jxTaydBnwIg` | correct | 9.000 | 0.947 | 1 | 2/2 |
| `XSC94UbpoRQ` | correct | 7.250 | 0.758 | 1 | 2/2 |
| `Hx0j-Oc18do` | undercount | 22.406 | 0.851 | 1 | 1/1 |
| `9lm02QThHjo` | undercount | 21.664 | 0.931 | 1 | 1/1 |
| `sVPA0POzHFQ` | undercount | 20.225 | 0.782 | 1 | 1/1 |
| `IZxOT44FKsU` | undercount | 19.485 | 0.983 | 1 | 1/1 |
| `HTwksOmzTsc` | undercount | 18.539 | 0.901 | 1 | 1/1 |
| `kmP3mlKeGRY` | undercount | 17.677 | 0.831 | 1 | 1/1 |
| `_5qHkZ0miYY` | undercount | 16.788 | 0.772 | 1 | 1/1 |
| `XJG-o9BK4To` | undercount | 13.207 | 0.894 | 1 | 3/1 |
| `lDvnabH6iyY` | undercount | 10.135 | 0.829 | 1 | 1/1 |
| `kkPMThxTwq0` | undercount | 9.448 | 0.848 | 1 | 1/1 |
| `hRYWiO-zi5s` | undercount | 7.790 | 0.859 | 1 | 1/1 |
| `kBbOVOaU6cc` | undercount | 6.422 | 0.967 | 1 | 1/1 |
| `LNdIO_4fRuI` | undercount | 5.870 | 0.737 | 2 | 1/1 |
| `sxOvp1cgE14` | undercount | 4.884 | 0.969 | 1 | 1/1 |
| `jE9ZB-zqEgs` | overcount | 35.496 | 0.631 | 4 | 10/10 |

The largest selected-case error is `jE9ZB-zqEgs`: ten clusters for two
reference speakers and 35.496 SpeakerError seconds. Its low AUC relative to the
other selected recordings is consistent with difficult embedding separation,
but does not identify recording conditions as the cause. `B5rLHPUJ8qc` and
`XJG-o9BK4To` are the only selected cases where the final cluster count changed
after the initial AHC stage; the summary flags temporal smoothing as a candidate,
not a proven cause.

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
