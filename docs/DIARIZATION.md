# Diarization configuration and experiments

## Overclustering and thresholds

Overclustering occurs when speech from one real speaker is split across several
predicted labels. It usually raises SpeakerError and the predicted speaker count.
Underclustering is the opposite failure: distinct speakers are assigned one
label. Speaker-count accuracy is therefore useful, but corpus DER remains the
primary selection metric.

AHC `threshold` is a **cosine distance** cutoff. A larger value generally permits
more AHC merges and produces fewer clusters. Centroid post-merge uses a **cosine
similarity** cutoff. A larger centroid threshold is stricter and permits fewer
post-merges. These parameters have opposite interpretations and must not be
interchanged.

Centroid post-merge is opt-in. It considers only clusters with at most
`max_small_cluster_size` embeddings, requires the best normalized-centroid
similarity to reach the configured threshold, and requires a margin over the
second-best target. Rejected fragments keep their original assignments. The
stage is skipped when `n_speakers` is supplied so it cannot violate an explicit
speaker-count constraint.

Adaptive AHC count estimation is also opt-in:

```powershell
python -m diarize.pipeline "data\audio\sample.wav" `
  --model "exps\exp1\model\model_0034.model" `
  --method ahc --count_estimator adaptive_ahc `
  --output "results\adaptive\sample.rttm"
```

`adaptive_ahc` preserves the original gap rule. The separate
`adaptive_ahc_robust` option uses a leave-one-out Tukey reference sample and
rejects tied evidence. Both cut the already fitted AHC tree deterministically,
report gap uncertainty fields, and fall back to the configured fixed threshold
when the hierarchy has no decisive gap. The repository default remains
fixed-threshold AHC. See
[`SPEAKER_CLUSTERING_IMPROVEMENTS.md`](SPEAKER_CLUSTERING_IMPROVEMENTS.md) for
the stopping rule and [`CLUSTERING_IMPROVEMENTS_V2.md`](CLUSTERING_IMPROVEMENTS_V2.md)
for the reused-test v2 measurements and limitations.

## Inference

The historical CLI remains valid:

```bash
python -m diarize.pipeline audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method ahc --threshold 0.65 --output results/audio.rttm
```

Enable or disable centroid merging explicitly:

```bash
python -m diarize.pipeline audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method ahc --threshold 0.75 \
  --centroid_merge --centroid_similarity_threshold 0.80 \
  --max_small_cluster_size 4 --output results/audio.rttm

python -m diarize.pipeline audio.wav \
  --model exps/exp1/model/model_0034.model \
  --no_centroid_merge --output results/audio.rttm
```

Automatic mode omits `--n_speakers`. Known-count mode passes application
knowledge explicitly:

```bash
python -m diarize.pipeline audio.wav \
  --model exps/exp1/model/model_0034.model \
  --n_speakers 2 --output results/audio.rttm
```

Using a reference annotation to supply that value is an **oracle experiment**,
not fully automatic diarization. Temporal smoothing may still eliminate a label
after clustering; diagnostics distinguish the cluster count from final RTTM
speaker count.

NME-SC is an additional backend; AHC and the existing spectral backend remain
available:

```bash
python -m diarize.pipeline audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method nme_sc --seed 42 --output results/audio.rttm
```

NME-SC excludes self-similarity from informative graph neighbors. With fewer
than eight embeddings, `floor(N/4)` does not provide two neighborhood sizes to
auto-tune, so the backend deterministically uses the existing AHC fallback.
Explicit speaker counts are preserved. This changes only NME-SC short-recording
handling; AHC and the original spectral backend are unchanged.

## Evaluation regions

Without UEM, scoring retains the historical full-audio behavior. Use an
author-provided UEM when the benchmark defines restricted evaluation regions:

```powershell
python -m eval.der `
  --hyp_dir results/hypothesis `
  --ref_dir data/reference `
  --uem_dir data/official_uem `
  --collar 0.25
```

UEM rows use `<recording-id> <channel> <start> <end>`. Multiple, adjacent, and
overlapping rows are normalized per recording. Missing IDs, invalid timestamps,
multiple channels, and intervals beyond known audio durations fail explicitly.
Do not derive UEM from model errors. The ViYT-Diar protocol evidence and open
questions are recorded in [`VIYT_EVALUATION_PROTOCOL.md`](VIYT_EVALUATION_PROTOCOL.md).

## YAML configuration

`configs/diarization_default.yaml` is loadable with `--config`. Precedence is:

```text
explicit CLI argument > YAML value > built-in default
```

Only explicitly supplied CLI arguments override YAML values. Unknown keys and
invalid types or ranges fail with an actionable error. Existing callers do not
need a YAML file.

```bash
python -m diarize.pipeline audio.wav \
  --model exps/exp1/model/model_0034.model \
  --config configs/diarization_default.yaml \
  --vad_min_silence_ms 200 --output results/audio.rttm
```

The effective configuration is logged before inference.

## Controlled experiment runner

The runner executes one-factor-at-a-time series rather than a Cartesian grid.
It loads the encoder once per invocation and writes the complete configuration,
model/data hashes, environment, RTTM files, corpus metrics, and per-recording
diagnostics.

```bash
python -m experiments.run_diarization_experiments \
  --audio_dir data/diarization/audio \
  --ref_rttm_dir data/diarization/reference_rttm \
  --model exps/exp1/model/model_0034.model \
  --output_dir results/ahc_sweep \
  --experiment ahc_threshold --device auto --seed 42
```

Add `--uem_dir data/official_uem` only when that UEM is part of the benchmark
protocol. The manifest records `full_audio` or `external_uem`, UEM paths and
hashes, collar, and overlap handling.

Supported series:

- `ahc_threshold`: distances 0.65, 0.70, 0.75, 0.80;
- `ahc_stability`: development-only distances 0.70, 0.72, 0.74, 0.75, 0.77;
- `centroid_merge`: AHC-only plus similarity 0.75, 0.80, 0.85;
- `window`: window/step 1.5/0.75, 2.0/1.0, 3.0/1.5 seconds;
- `vad_silence`: 100, 200, 300 ms;
- `speaker_count`: oracle known count and automatic count;
- `clustering_count`: fixed AHC, legacy adaptive AHC, robust adaptive AHC,
  spectral and NME-SC under the same segmentation settings, all with automatic
  count;
- `clustering`: AHC, spectral with silhouette count selection, and NME-SC;
- `vad_threshold` and `all` remain available.

Use a new output directory for every invocation. Inspect a plan without loading
audio or a model:

```bash
python -m experiments.run_diarization_experiments \
  --audio_dir data/diarization/audio \
  --ref_rttm_dir data/diarization/reference_rttm \
  --output_dir results/dry_run --experiment all --dry_run
```

Per-recording diagnostics include the checkpoint, clustering and segmentation
configuration, embedding-window count, cluster counts and sizes before/after
post-merge, final RTTM speaker count, singleton/small-cluster counts, reference
and predicted counts, DER components, and centroid merge attempts/scores.
Reference-dependent fields are null when references are unavailable.
Compatible configurations reuse an in-memory embedding cache and report cache
hits. VAD or window changes always use a different cache entry.

See [`DIARIZATION_ERROR_ANALYSIS.md`](DIARIZATION_ERROR_ANALYSIS.md) for the
reference-coverage audit, SpeakerError count split, centroid rejection evidence,
and the development-only AHC stability workflow.

The optional embedding/pipeline diagnostic workflow is documented in
[`EMBEDDING_ERROR_ANALYSIS.md`](EMBEDDING_ERROR_ANALYSIS.md). It records the
actual AHC merge hierarchy, intermediate labels, reference-overlap classes,
counterfactual postprocessing traces, and four reproducible plots. Diagnostic
artifacts are provenance-bound and must not be used to tune production settings
on the previously used ViYT-Diar test set.

## Corrected-encoder frozen reproduction

The checkpoint-compatible `Bottle2neck` order is
`Conv1d -> ReLU -> BatchNorm1d`. With checkpoint SHA-256
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`
and encoder-source SHA-256
`96cd2147b769f3730d11eaabefedb3f3b52c3913f87f74d01b208d4bf92b14d8`,
the frozen AHC 0.75/VAD 200 ms configuration evaluated exactly 100 recordings
without failures. Every generated RTTM had the same SHA-256 as its preserved
historical counterpart.

Scoring used full audio, a 0.25-second collar, included overlap, and pooled
component seconds. No UEM or oracle speaker count was used.

| Reference mode | DER | Miss | FA | SpeakerError | Count accuracy | Count MAE | Over | Under |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Published reference | 15.1240 | 1.1813 | 4.9282 | 9.0144 | 0.60 | 0.65 | 0.26 | 0.14 |
| Manual-audit sensitivity | 12.8687 | 1.7092 | 2.3043 | 8.8552 | 0.61 | 0.62 | 0.25 | 0.14 |

The sensitivity row substitutes manually audited references only for
`C4HFERLpRfI`, `Uaq5N3FbQb4`, and `ed0Vt0NxJUI`; the hypotheses are identical.
Those broad reviewed turns may include pauses and are not an official ViYT-Diar
reference. See [`CORRECTED_VIYT_DIAR_EVALUATION.md`](CORRECTED_VIYT_DIAR_EVALUATION.md)
for provenance, commands, and diagnostic scope.

## ViYT-Diar exploratory measurements

The historical baseline and the new runs use all 100 ViYT-Diar test recordings
(104.32 minutes), dataset revision
`c66b118709d68c7538408d086232d850356db374`, epoch-34 checkpoint SHA256
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`,
collar 0.25 seconds, overlap included, and seed 42. The new runs were executed
locally on an RTX 5060 Laptop GPU. The scorer and reference annotations were not
changed.

Only this previously evaluated test corpus was available. All new sweeps below
are therefore **exploratory test-set analysis**, not unbiased held-out results,
and must not be used to claim a new production default. A speaker-disjoint
development set is required for parameter selection, followed by one untouched
test evaluation.

### AHC threshold sweep

All other settings were fixed to window/step 1.5/0.75 seconds, VAD minimum
silence 100 ms, and centroid merge disabled.

| AHC distance | DER | Miss | FA | SpeakerError | Count accuracy | Count MAE | Over | Under | Avg pred/ref |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.65 (historical baseline, reproduced) | 17.6331 | 1.6532 | 4.5967 | 11.3831 | 0.32 | 1.74 | 0.66 | 0.02 | 3.67/1.97 |
| 0.70 | 15.8009 | 1.6532 | 4.5967 | 9.5509 | 0.43 | 1.06 | 0.49 | 0.08 | 2.87/1.97 |
| 0.75 | **15.6776** | 1.6532 | 4.5967 | **9.4276** | 0.59 | 0.65 | 0.26 | 0.15 | 2.32/1.97 |
| 0.80 | 17.1644 | 1.6532 | 4.5967 | 10.9145 | 0.54 | 0.58 | 0.20 | 0.26 | 2.03/1.97 |

Threshold 0.75 has the lowest exploratory DER. Threshold 0.80 improves count
MAE slightly but increases underclustering and DER, demonstrating why count
accuracy alone is not a safe selection criterion.

### Centroid merge

With AHC distance 0.75, the AHC-only configuration and centroid similarities
0.75, 0.80, and 0.85 all produced DER 15.6776, SpeakerError 9.4276, count
accuracy 0.59, and count MAE 0.65. Across the corpus each enabled configuration
recorded 95 candidate attempts, zero accepted merges, and 95 rejections. The
safety checks preserved the baseline but this sweep provides no evidence for
enabling centroid merge by default.

### Window and VAD series

| Series | Setting | DER | SpeakerError | Count accuracy | Count MAE | Over | Under |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Window | 1.5/0.75 s | **15.6776** | **9.4276** | 0.59 | 0.65 | 0.26 | 0.15 |
| Window | 2.0/1.0 s | 16.9840 | 10.7340 | 0.56 | 0.57 | 0.20 | 0.24 |
| Window | 3.0/1.5 s | 18.0560 | 11.8061 | 0.59 | **0.43** | 0.13 | 0.28 |
| VAD min silence | 100 ms | 15.6776 | 9.4276 | 0.59 | 0.65 | 0.26 | 0.15 |
| VAD min silence | 200 ms | **15.1240** | **9.0144** | **0.60** | 0.65 | 0.26 | **0.14** |
| VAD min silence | 300 ms | 15.4234 | 9.0397 | 0.57 | 0.67 | 0.26 | 0.17 |

Longer windows reduce speaker-count MAE but worsen DER and underclustering. The
200 ms VAD setting is the best measured exploratory combination with AHC 0.75,
but still requires development/held-out confirmation.

### Known speaker count

At AHC 0.75 and VAD 200 ms, fully automatic mode produced DER 15.1240,
SpeakerError 9.0144, count accuracy 0.60, and count MAE 0.65. Supplying the
reference count as an oracle produced DER 13.0825, SpeakerError 6.9730, count
accuracy 0.91, and count MAE 0.09. The oracle result is an upper-bound diagnostic
and must never be reported as automatic performance.

The validated paired values before display rounding are automatic DER
`15.1239746592`, SpeakerError `9.0144201731`, and oracle-count DER
`13.0825466601`, SpeakerError `6.9729921740`. Automatic RTTMs reproduced the
frozen baseline for all 100 recordings. SpeakerError time changed from
`399.252003` to `308.836402` seconds; 28 recordings improved, 10 worsened and 62
were unchanged. Nine requested two-speaker recordings still lost one identity
during postprocessing. These are reused-test diagnostics, not evidence for a
production setting.

### Smoothing counterfactual

For `XJG-o9BK4To`, the trace changes from three initial clusters to one after
temporal smoothing. Disabling smoothing does not improve the measured result:
current smoothing gives DER `37.35515`, versus `37.78454` without smoothing.
No smoothing change is promoted. This is a single-recording, reused-test
diagnostic; the exact PowerShell workflow is in
[`SPEAKER_CLUSTERING_IMPROVEMENTS.md`](SPEAKER_CLUSTERING_IMPROVEMENTS.md).

### Backend comparison

Backend results are recorded by `--experiment clustering` with identical
segmentation settings. Runtime is wall-clock time for the 100-recording series;
memory is Python-tracked peak memory and does not include all CUDA allocations.

| Backend | DER | SpeakerError | Count accuracy | Count MAE | Over | Under | Avg pred/ref | Runtime (s) | Peak Python MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AHC | **15.1240** | **9.0144** | 0.60 | **0.65** | 0.26 | 0.14 | 2.34/1.97 | **123.64** | 24.34 |
| Spectral | 16.6488 | 10.5393 | **0.65** | 0.76 | 0.22 | **0.13** | 2.47/1.97 | 179.70 | 11.19 |
| NME-SC | 17.3243 | 11.2148 | 0.48 | 0.73 | **0.14** | 0.38 | 1.94/1.97 | 132.92 | **10.88** |

The first configuration also bears one-time Python allocation overhead, so peak
memory values are indicative rather than a precise GPU-memory comparison. All
three methods used the same frozen encoder and deterministic segmentation
configuration. AHC had the best exploratory DER. NME-SC reduced overclustering
but underclustered 38% of recordings and did not improve end-to-end quality.

## Current recommendation

Keep the repository defaults unchanged until a separate development set exists:
AHC 0.65, window/step 1.5/0.75 seconds, VAD minimum silence 100 ms, and centroid
merge disabled. For a future development-set validation, the measured candidate
is AHC distance 0.75 with VAD minimum silence 200 ms, the original window, and
centroid merge disabled. This is a hypothesis derived from exploratory test-set
analysis, not a validated production default.

PLDA and ECAPA fine-tuning were not evaluated. See
[`PLDA_FEASIBILITY.md`](PLDA_FEASIBILITY.md) and
[`EMBEDDING_DISCRIMINABILITY.md`](EMBEDDING_DISCRIMINABILITY.md) for the data
requirements and leakage-safe evaluation plans.

## Kaggle execution after ViYT export

`kaggle/ViYT_Diar_DER.ipynb` defines `AUDIO`, `REFERENCE`, `MODEL`, and `RESULTS`
after exporting ViYT-Diar. The following complete notebook cell runs the same
sequential ablations in separate output directories. Each invocation loads the
checkpoint once and reuses it for every configuration in that series.

```python
import subprocess
import sys

common = [
    sys.executable, "-m", "experiments.run_diarization_experiments",
    "--audio_dir", str(AUDIO),
    "--ref_rttm_dir", str(REFERENCE),
    "--model", str(MODEL),
    "--device", "cuda", "--seed", "42", "--collar", "0.25",
]

runs = [
    ("ahc_threshold", ["--experiment", "ahc_threshold",
                       "--vad_min_silence_ms", "100"]),
    ("centroid_merge", ["--experiment", "centroid_merge",
                        "--threshold", "0.75",
                        "--vad_min_silence_ms", "100"]),
    ("window", ["--experiment", "window", "--threshold", "0.75",
                "--vad_min_silence_ms", "100"]),
    ("vad_silence", ["--experiment", "vad_silence", "--threshold", "0.75"]),
    ("speaker_count", ["--experiment", "speaker_count",
                       "--threshold", "0.75",
                       "--vad_min_silence_ms", "200"]),
    ("clustering", ["--experiment", "clustering", "--threshold", "0.75",
                    "--vad_min_silence_ms", "200"]),
]

for name, arguments in runs:
    output = RESULTS / f"overclustering_{name}"
    subprocess.run(common + ["--output_dir", str(output)] + arguments,
                   check=True)
```

Do not rerun into an existing output directory. Select parameters on a new
development split, then run exactly one final configuration on a held-out test
split. Generated data, checkpoints, RTTM outputs, and experiment directories
must remain outside Git.
