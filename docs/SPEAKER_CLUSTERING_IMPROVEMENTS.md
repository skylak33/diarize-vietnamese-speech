# Speaker clustering and count estimation

## Status and architecture

The production path remains unchanged:

```text
audio -> VAD -> overlapping ECAPA windows -> clustering -> temporal smoothing
      -> short-segment merge -> adjacent-segment merge -> RTTM
```

The default is still automatic AHC with cosine distance `0.65` and
`count_estimator: backend_default`. Adaptive AHC is an **opt-in experiment**;
there is not yet independent-corpus evidence for promoting it. A known speaker
count bypasses automatic count estimation and is an oracle diagnostic when the
value comes from reference RTTM. Diagnostics report both the clustering count
and the final RTTM count because postprocessing can remove a speaker identity.

The relevant interfaces are:

- `diarize/cluster.py`: fixed AHC, adaptive hierarchy stopping, spectral and
  NME-SC backends;
- `diarize/pipeline.py`: segmentation, embedding cache, clustering,
  postprocessing, RTTM and diagnostic artifacts;
- `diarize/config.py` and `configs/diarization_default.yaml`: validated opt-in
  configuration with CLI-over-YAML precedence;
- `experiments/run_diarization_experiments.py`: paired, one-factor comparisons
  and oracle-count reports.

## Independent-data boundary

No independent diarization corpus is currently available locally. ViYT-Diar's
100 test recordings have already been reused for exploration, while the local
VLSP material is verification/train speech without authoritative conversational
speaker turns. Consequently, the adaptive AHC and backend comparison has **no
unbiased empirical DER result**. Existing ViYT measurements are labeled reused-
test diagnostics and must not select production settings.

The acquisition and leakage-safe split plan is documented in
[`DIARIZATION_DEVELOPMENT_DATA.md`](DIARIZATION_DEVELOPMENT_DATA.md). In short,
obtain the official VoxConverse v0.3 development WAV and RTTM data only after
storage/license approval, record the exact annotation commit, group clips by
source/conversation, split those groups into development and validation, and
leave the official test set untouched. Tune on development and compare one
frozen choice on validation. Domain mismatch still requires later confirmation
on newly acquired Vietnamese conversational data.

## Adaptive AHC rule

For `N` embedding windows, full average-linkage AHC produces ordered merge
distances

```text
d[1] <= d[2] <= ... <= d[N-1].
```

For each adjacent gap `g[i] = d[i+1] - d[i]`, stopping before merge `i+1`
implies `k = N - i` clusters. Only `2 <= k <= max_speakers` is eligible. The
estimator computes Tukey's upper fence over eligible gaps,
`Q3 + 1.5 * (Q3 - Q1)`, and selects the largest gap only when it exceeds that
fence. A jump after numerically flat, near-zero merges is also accepted. If no
gap is decisive, the existing fixed-distance threshold supplies a conservative
fallback. The chosen count is then passed to a second AHC fit.

The gap scan is `O(N)` time and memory. The hierarchy dominates: the current
scikit-learn implementation requires quadratic pairwise work/storage in the
number of windows in the typical case. Adaptive mode constructs the hierarchy
and then refits AHC at the selected count, so it can approach twice the fixed-AHC
clustering cost. This cost is reported separately from embedding extraction.

Enable it explicitly:

```powershell
python -m diarize.pipeline "data\audio\sample.wav" `
  --model "exps\exp1\model\model_0034.model" `
  --method ahc --count_estimator adaptive_ahc `
  --max_speakers 10 --output "results\adaptive\sample.rttm"
```

Or in YAML:

```yaml
clustering:
  method: ahc
  count_estimator: adaptive_ahc
```

It is rejected with spectral/NME-SC or with `--n_speakers`, because those cases
have their own estimator or an explicit count.

## Fair automatic comparison

`clustering_count` compares fixed-threshold AHC, adaptive AHC, spectral
silhouette selection and NME-SC under identical VAD/window/model/seed settings.
All four configurations are automatic; reference speaker counts are never
passed to them. The runner loads the encoder once and reuses only compatible
embeddings. Backend-specific clustering time and count diagnostics remain
separate. Use a new output directory for every run.

Inspect the exact plan without model/data access:

```powershell
python -m experiments.run_diarization_experiments `
  --audio_dir "data\voxconverse\development\audio" `
  --ref_rttm_dir "data\voxconverse\development\rttm" `
  --output_dir "results\clustering_count_plan" `
  --experiment clustering_count --dry_run
```

Run it after the independent manifest has been frozen:

```powershell
python -m experiments.run_diarization_experiments `
  --audio_dir "data\voxconverse\development\audio" `
  --ref_rttm_dir "data\voxconverse\development\rttm" `
  --model "exps\exp1\model\model_0034.model" `
  --output_dir "results\voxconverse_clustering_count" `
  --experiment clustering_count --device auto --seed 42 --collar 0.25
```

Until that run exists, backend/adaptive comparison tables must say
"not measured on independent data" rather than reuse ViYT numbers.

## Oracle speaker-count diagnostic

The paired ViYT run used the same 100 recordings, checkpoint, segmentation and
scorer in both conditions. The automatic RTTMs exactly matched the frozen
baseline (100/100). The result is diagnostic on a previously used test set, not
a production estimate:

| Condition | DER | SpeakerError | Count accuracy | Count MAE |
| --- | ---: | ---: | ---: | ---: |
| Automatic | 15.1239746592 | 9.0144201731 | 0.60 | 0.65 |
| Reference-count oracle | 13.0825466601 | 6.9729921740 | 0.91 | 0.09 |

SpeakerError fell from `399.252003` to `308.836402` seconds. Per recording, 28
improved, 10 worsened and 62 were unchanged. Nine requested two-speaker cases
still ended with one RTTM identity because postprocessing removed a label. Thus
the gap isolates a substantial count-estimation contribution, but it also shows
that correct cluster count alone does not guarantee correct assignments or final
speaker preservation.

Reproduce the paired diagnostic only as labeled analysis:

```powershell
python -m experiments.run_diarization_experiments `
  --audio_dir "data\viyt\audio" `
  --ref_rttm_dir "data\viyt\reference_rttm" `
  --model "exps\exp1\model\model_0034.model" `
  --output_dir "results\viyt_oracle_diagnostic" `
  --experiment speaker_count --method ahc --threshold 0.75 `
  --vad_min_silence_ms 200 --device auto --seed 42 --collar 0.25
```

`oracle_count_diagnostics.json` labels the condition as non-automatic and stores
paired automatic-minus-oracle deltas. Do not mix its row into an automatic
leaderboard.

## Smoothing investigation: `XJG-o9BK4To`

The corrected diagnostic trace confirms the reported mechanism: initial AHC
produces three clusters, temporal smoothing reduces them to one, and the final
RTTM retains one speaker. With the reference count, clustering starts with two
clusters and smoothing still reduces them to one. However, the measured
counterfactual is not an improvement: current smoothing gives DER `37.35515`
and `13.207270` speaker-error seconds, while disabling smoothing gives DER
`37.78454` and `13.407274` seconds. Therefore no smoothing change is promoted.

Regenerate the trace and counterfactual outside Git:

```powershell
$audio = "data\viyt\audio\XJG-o9BK4To.wav"
$model = "exps\exp1\model\model_0034.model"
$references = "data\viyt\reference_rttm"
$artifacts = "results\xjg_smoothing_diagnostic"

python -m diarize.pipeline $audio `
  --model $model --method ahc --threshold 0.75 `
  --vad_min_silence_ms 200 `
  --output "$artifacts\XJG-o9BK4To.rttm" `
  --diagnostic_output_dir $artifacts --diagnostic_counterfactuals `
  --evaluation_protocol "ViYT-Diar full audio; collar=0.25; include overlap" `
  --reference_source published_reference

python -m experiments.analyze_embedding_diagnostics `
  --artifacts $artifacts --reference_rttm $references `
  --recording_id XJG-o9BK4To --output_dir $artifacts

python -m experiments.visualize_embedding_diagnostics `
  "$artifacts\XJG-o9BK4To" `
  --output_dir "$artifacts\XJG-o9BK4To" --format png

python -m experiments.analyze_smoothing_counterfactuals `
  --artifacts "$artifacts\XJG-o9BK4To" `
  --reference-rttm $references --wav $audio `
  --output "$artifacts\XJG-o9BK4To\smoothing_counterfactuals.json"
```

The no-smoothing RTTM is
`results\xjg_smoothing_diagnostic\counterfactuals\XJG-o9BK4To\without_temporal_smoothing.rttm`.
This single reused-test example explains a failure mode; it cannot justify a
global smoothing policy.

## Verification

Run the focused integration checks and the CPU suite from the repository root:

```powershell
python -m pytest -q tests\test_clustering_count_integration.py
python -m pytest -q
python -m compileall -q models train diarize eval experiments utils tests
```

## Limitations

- There is no independent empirical clustering-count comparison yet.
- Oracle count uses reference information and remains an upper-bound diagnostic.
- AHC windows do not model overlapping speakers; one label is emitted per time.
- Speaker count can be correct while cluster assignments are wrong, and temporal
  postprocessing can change the final count.
- Adaptive stopping is heuristic, sensitive to embedding/window quality, and
  adds an extra AHC fit.
- VoxConverse/AMI are English and domain-mismatched; Vietnamese validation is
  still required before changing defaults.
