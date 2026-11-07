# Corrected ViYT-Diar evaluation

## Outcome

The checkpoint-compatible ECAPA encoder reproduced the preserved historical
diarization output exactly: 100 of 100 corrected RTTM files have the same
SHA-256 as the corresponding historical RTTM. This establishes a reproducible
baseline for the frozen configuration. It is a historical exploratory
reproduction, not a new held-out quality estimate, because ViYT-Diar was already
used for parameter exploration.

## Provenance and protocol

| Item | Recorded value |
| --- | --- |
| Git commit | `eb08eb579018ce6107a7600c6aa22cad6654b3fb` |
| Git state during run | clean |
| ECAPA `Bottle2neck` order | `Conv1d -> ReLU -> BatchNorm1d` |
| Encoder source SHA-256 | `96cd2147b769f3730d11eaabefedb3f3b52c3913f87f74d01b208d4bf92b14d8` |
| Checkpoint SHA-256 | `58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325` |
| Python / PyTorch | 3.14.3 / 2.11.0+cu130 |
| Platform / device request | Windows 11 / CUDA |
| Recordings | 100, no failures |
| ViYT-Diar revision used by the historical protocol | `c66b118709d68c7538408d086232d850356db374` |
| Scoring | full audio, 0.25 s collar, overlap included, no UEM |
| Speaker count | automatic, maximum 10; no oracle count |
| Seed | 42 |

The run manifest records per-recording audio and reference hashes. It records
the CUDA device request but not the GPU model or CUDA runtime version, so those
two fields cannot be independently recovered from this artifact alone.

The frozen inference configuration was: 1.5/0.75-second embedding window/step,
minimum segment 0.5 seconds; Silero VAD threshold 0.5, minimum speech 250 ms,
minimum silence 200 ms, padding 30 ms; AHC cosine-distance threshold 0.75;
centroid merge disabled.

## Historical reproduction controls

| Recording | Expected speakers | Corrected speakers | Exact historical RTTM hash |
| --- | ---: | ---: | --- |
| `98EV6H0ZRlU` | 2 | 2 | yes |
| `B5rLHPUJ8qc` | 2 | 2 | yes |
| `jxTaydBnwIg` | 2 | 2 | yes |
| `XSC94UbpoRQ` | 2 | 2 | yes |
| `jE9ZB-zqEgs` | 10 | 10 | yes |

The complete comparison extended the same check to all 100 recordings and found
100 exact matches. This is bytewise reproducibility, not merely equality after
speaker-label permutation.

## Published-reference result

Corpus rates are pooled component seconds divided by pooled scored reference
speaker-time; they are not means of per-recording percentages.

| Metric | Historical run | Corrected reproduction |
| --- | ---: | ---: |
| DER | 15.123975% | 15.123975% |
| Miss | 1.181326% | 1.181326% |
| False alarm | 4.928228% | 4.928228% |
| SpeakerError | 9.014420% | 9.014420% |
| Speaker-count accuracy | 60% | 60% |
| Speaker-count MAE | 0.65 | 0.65 |
| Overcount rate | 26% | 26% |
| Undercount rate | 14% | 14% |

The pooled denominator was 4,429.036980 speaker-seconds. Error components were
52.321377 miss seconds, 218.273051 false-alarm seconds, and 399.252003
SpeakerError seconds. Average predicted/reference speaker counts were 2.34/1.97.

## Manual-reference sensitivity

The same frozen hypotheses were rescored after substituting manually audited
references for `C4HFERLpRfI`, `Uaq5N3FbQb4`, and `ed0Vt0NxJUI` only.

| Metric | Published reference | Manual-audit sensitivity |
| --- | ---: | ---: |
| DER | 15.123975% | 12.868721% |
| Miss | 1.181326% | 1.709206% |
| False alarm | 4.928228% | 2.304287% |
| SpeakerError | 9.014420% | 8.855228% |
| Speaker-count accuracy | 60% | 61% |
| Speaker-count MAE | 0.65 | 0.62 |

The hypotheses are identical across the two rows. The manual annotations contain
broad reviewed turns that may include pauses, so this sensitivity result is not
the official benchmark and should not be interpreted as a more accurate VAD
measurement without further annotation work.

## Architecture and embedding geometry

The wrong-order graph (`Conv -> BatchNorm -> ReLU`) and corrected graph use the
same tensor names and shapes, so strict checkpoint loading cannot distinguish
them. On the same 1,317 windows from 19 selected recordings, global descriptive
geometry changed as follows:

| Metric | Wrong-order artifacts | Corrected artifacts |
| --- | ---: | ---: |
| Normalized centroid norm | 0.998002 | 0.280187 |
| Effective rank | 20.574 | 66.144 |
| Mean pairwise cosine | 0.996004 | 0.077805 |
| Median pairwise cosine | 0.996662 | 0.054180 |
| Pairwise cosine standard deviation | 0.002532 | 0.165994 |

These measurements establish that the near-collinear wrong-order representation
was not reused. They do not by themselves measure speaker discriminability or
prove that a particular remaining diarization error is caused by AHC.

Reference-aware statistics for the 19 corrected diagnostic recordings give an
unweighted mean same-speaker cosine of 0.4973, mean different-speaker cosine of
0.3075, and per-recording diagnostic AUC mean/median of 0.8555/0.8508 (range
0.6307-0.9834). Median KS, Wasserstein distance, and distribution overlap were
0.5905, 0.1801, and 0.4131. Overlapping windows and repeated pairs make these
descriptive statistics, not independent observations.

## Remaining SpeakerError

Corpus-wide speaker counts were correct for 60 recordings, overcounted for 26,
and undercounted for 14. Count-error recordings account for 326.930 of the
399.252 SpeakerError seconds (81.9%). The pooled breakdown is:

| Count group | Recordings | Reference speaker-s | Miss s | FA s | SpeakerError s | SpeakerError rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct count | 60 | 2,698.788 | 24.240 | 66.842 | 72.322 | 2.680% |
| Overcount | 26 | 1,061.024 | 16.217 | 144.375 | 132.389 | 12.477% |
| Undercount | 14 | 669.224 | 11.864 | 7.056 | 194.540 | 29.070% |

Across all recordings, long speaker-confusion intervals contributed 228.597
speaker-seconds and boundary-adjacent confusion contributed 67.392
speaker-seconds. These are temporal diagnostic categories, not inferred
recording conditions.

The ten largest SpeakerError contributors are:

| Rank | Recording | Count status | Ref/pred | SpeakerError s |
| ---: | --- | --- | ---: | ---: |
| 1 | `jE9ZB-zqEgs` | overcount | 2/10 | 35.496 |
| 2 | `Hx0j-Oc18do` | undercount | 2/1 | 22.406 |
| 3 | `9lm02QThHjo` | undercount | 2/1 | 21.664 |
| 4 | `sVPA0POzHFQ` | undercount | 2/1 | 20.225 |
| 5 | `IZxOT44FKsU` | undercount | 2/1 | 19.485 |
| 6 | `HTwksOmzTsc` | undercount | 2/1 | 18.539 |
| 7 | `kmP3mlKeGRY` | undercount | 2/1 | 17.677 |
| 8 | `_5qHkZ0miYY` | undercount | 2/1 | 16.788 |
| 9 | `V-cuzXOX9sM` | overcount | 2/5 | 16.378 |
| 10 | `98EV6H0ZRlU` | correct | 2/2 | 15.312 |

Among the original 19 diagnostic recordings, corrected count relations are 4
correct, 14 undercount, and 1 overcount. Reference-aware AHC traces contain at
least one merge joining reference-distinct speakers in each selected recording.
The automated evidence nevertheless marks 18 conclusions low-confidence and
one (`jxTaydBnwIg`) unknown-confidence because mixed/overlapping windows and
reference reliability prevent clean causal attribution. It confirms no physical
recording cause. AHC overmerging, fragmentation, mixed windows, and smoothing
remain candidates for targeted review.

`jE9ZB-zqEgs` is the clearest engineering priority in this set: its AUC is the
lowest of the 19 (0.6307), it remains at ten clusters after postprocessing, and
it contributes the most SpeakerError. `B5rLHPUJ8qc` changes from three initial
clusters to two final clusters, and `XJG-o9BK4To` from three to one; smoothing is
therefore a candidate mechanism in those two cases, but the current artifacts
do not prove it caused their reference-aligned errors. See
[`EMBEDDING_ERROR_ANALYSIS.md`](EMBEDDING_ERROR_ANALYSIS.md) for the full ledger.

### Original 19-recording root-cause ledger

`Embedding` reports diagnostic AUC and mean same/different-reference-speaker
cosine. `AHC` is the count of traced merges whose two inputs contain distinct
reference speakers. `Post` gives initial/after-smoothing/final cluster counts.
Every conclusion below is an automated candidate, not a confirmed physical cause.

| Recording | Ref/pred | SpeakerError s | Embedding | AHC | Post | Conclusion |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `jE9ZB-zqEgs` | 2/10 | 35.496 | AUC .631; .231/.167 | 4 | 10/10/10 | Low confidence: fragmentation/overlap; smoothing changed one label only |
| `Hx0j-Oc18do` | 2/1 | 22.406 | AUC .851; .493/.313 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `9lm02QThHjo` | 2/1 | 21.664 | AUC .931; .552/.316 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `sVPA0POzHFQ` | 2/1 | 20.225 | AUC .782; .493/.384 | 1 | 1/1/1 | Low confidence: overmerging |
| `IZxOT44FKsU` | 2/1 | 19.485 | AUC .983; .637/.339 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `HTwksOmzTsc` | 2/1 | 18.539 | AUC .901; .524/.324 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `kmP3mlKeGRY` | 2/1 | 17.677 | AUC .831; .452/.283 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `_5qHkZ0miYY` | 2/1 | 16.788 | AUC .772; .425/.292 | 1 | 1/1/1 | Low confidence: overmerging |
| `98EV6H0ZRlU` | 2/2 | 15.312 | AUC .940; .584/.285 | 1 | 2/2/2 | Low confidence: mixed windows |
| `XJG-o9BK4To` | 2/1 | 13.207 | AUC .894; .472/.243 | 1 | 3/1/1 | Low confidence: fragmentation, smoothing, mixed windows |
| `B5rLHPUJ8qc` | 2/2 | 11.810 | AUC .825; .518/.319 | 1 | 3/2/2 | Low confidence: fragmentation, smoothing, mixed windows |
| `lDvnabH6iyY` | 2/1 | 10.135 | AUC .829; .493/.350 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `kkPMThxTwq0` | 2/1 | 9.448 | AUC .848; .574/.419 | 1 | 1/1/1 | Low confidence: overmerging |
| `jxTaydBnwIg` | 2/2 | 9.000 | AUC .947; .481/.253 | 1 | 2/2/2 | Unknown: no automated candidate; inspect reliable reference evidence |
| `hRYWiO-zi5s` | 2/1 | 7.790 | AUC .859; .532/.378 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `XSC94UbpoRQ` | 2/2 | 7.250 | AUC .758; .411/.310 | 1 | 2/2/2 | Low confidence: mixed windows |
| `kBbOVOaU6cc` | 2/1 | 6.422 | AUC .967; .634/.337 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `LNdIO_4fRuI` | 2/1 | 5.870 | AUC .737; .417/.322 | 2 | 1/1/1 | Low confidence: overmerging, mixed windows |
| `sxOvp1cgE14` | 2/1 | 4.884 | AUC .969; .538/.230 | 1 | 1/1/1 | Low confidence: overmerging, mixed windows |

## Reproduction on Windows PowerShell

Run from the repository root and choose a new output directory; the runner does
not overwrite an existing experiment safely.

```powershell
$audio = "C:\path\to\viyt-data\audio"
$reference = "C:\path\to\published_reference_rttm"
$output = "results\viyt_corrected_reproduction"
$diagnostics = "results\embedding_analysis_corrected_reproduction"
$ids = @(
  "98EV6H0ZRlU", "B5rLHPUJ8qc", "jxTaydBnwIg", "XSC94UbpoRQ",
  "Hx0j-Oc18do", "9lm02QThHjo", "sVPA0POzHFQ", "IZxOT44FKsU",
  "HTwksOmzTsc", "kmP3mlKeGRY", "_5qHkZ0miYY", "XJG-o9BK4To",
  "lDvnabH6iyY", "kkPMThxTwq0", "hRYWiO-zi5s", "kBbOVOaU6cc",
  "LNdIO_4fRuI", "sxOvp1cgE14", "jE9ZB-zqEgs"
)

python -m experiments.run_diarization_experiments `
  --audio_dir $audio --ref_rttm_dir $reference `
  --output_dir $output `
  --model exps\exp1\model\model_0034.model `
  --experiment baseline --method ahc --threshold 0.75 `
  --vad_threshold 0.5 --vad_min_speech_ms 250 `
  --vad_min_silence_ms 200 --vad_speech_pad_ms 30 `
  --window_sec 1.5 --step_sec 0.75 --min_segment_sec 0.5 `
  --speaker_mode automatic --max_speakers 10 `
  --collar 0.25 --seed 42 --device cuda `
  --diagnostic_output_dir $diagnostics --diagnostic_ids $ids
```

The absence of `--skip_overlap`, `--uem_dir`, and `--centroid_merge` is
intentional. Before comparing results, verify the checkpoint, encoder source,
audio, reference, configuration, and environment hashes in `manifest.json`.
Manual-reference sensitivity must reuse these RTTMs and change only the explicit
reference set; it must not rerun inference.

## Limitations and next experiments

- ViYT-Diar has been reused for exploratory parameter selection. The reproduced
  numbers cannot support an independent generalization or production-default claim.
- The historical run did not preserve its exact dirty encoder-source hash or
  embeddings. Exact RTTM reproduction strongly supports compatibility but does
  not reconstruct that source snapshot.
- Wrong-order DER is available only for the selected 19 recordings and must not
  be compared as a full-corpus metric.
- Pairwise embedding samples are dependent because windows overlap and pairs
  repeat; descriptive AUC/KS/distribution statistics are not independent-sample
  significance tests.
- Manual-audit references are a sensitivity scenario, not an official replacement.

The next parameter or model comparison should use a speaker-disjoint development
set, freeze the selected configuration, and evaluate once on an untouched test
set. Do not introduce test-estimated whitening, PLDA, retraining, or threshold
changes based on this reproduction alone.

## Authoritative artifacts

- `results/viyt_corrected_20261103/manifest.json`
- `results/viyt_corrected_20261103/corpus_results.json`
- `results/viyt_corrected_20261103/reference_sensitivity.json`
- `results/viyt_corrected_20261103/reproducibility_validation.json`
- `results/viyt_corrected_20261103/historical_comparison.json`
- `results/viyt_corrected_20261103/speaker_error_summary.json`
- `results/viyt_corrected_20261103/speaker_error_recordings.csv`
- `results/viyt_corrected_20261103/top_error_recordings.csv`
- `results/embedding_analysis_corrected/summary.json`
- `results/embedding_analysis_corrected/` for the isolated corrected diagnostics
