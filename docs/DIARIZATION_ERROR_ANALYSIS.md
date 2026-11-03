# Diarization error analysis

## Scope and protocol

This report diagnoses the existing 100-recording ViYT-Diar test run using the
exploratory candidate AHC 0.75, VAD minimum silence 200 ms, and window/step
1.5/0.75 seconds. It does not alter reference annotations, UEM regions, or DER
scoring. Because this test set was already used for parameter selection, the
findings must not be treated as held-out validation.

## Suspected partial references

The source Parquet rows and exported RTTM files contain speech annotations only
near the end of three recordings. Silero VAD independently detects substantial
speech-like audio before the first reference segment.

| Recording | Audio s | Reference speech s | Coverage | First reference s | VAD speech before reference s | VAD speech outside reference s | FalseAlarm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `C4HFERLpRfI` | 62.710 | 4.344 | 6.9% | 57.415 | 50.115 | 50.756 | 1399.91% |
| `Uaq5N3FbQb4` | 60.054 | 11.210 | 18.7% | 43.122 | 40.822 | 43.550 | 477.18% |
| `ed0Vt0NxJUI` | 62.749 | 19.671 | 31.3% | 28.381 | 22.200 | 24.406 | 140.88% |

Manual listening confirms speech in the unannotated leading regions:

| Recording | Manual observation (approximate) |
| --- | --- |
| `C4HFERLpRfI` | Speaker 1 from 0:00; speaker 2 from 0:12; speaker 1 laughs near 0:44; speaker 2 resumes near 0:45. |
| `Uaq5N3FbQb4` | Speaker 1 from 0:00 over background music; speaker 2 begins near 0:36. |
| `ed0Vt0NxJUI` | Speaker 1 from 0:00; speaker 2 begins near 0:09. |

The review confirms at least two speakers in every recording, so their current
one-speaker references and full-audio speaker-count classifications are not
reliable. Background music in `Uaq5N3FbQb4` may account for some VAD activity,
but not the confirmed speech. These approximate change points are audit
evidence, not sufficiently precise segment boundaries for rewriting RTTM.

Before rescoring, either complete the references over the entire recordings or,
if the dataset intentionally annotates only selected regions, obtain the
intended UEM and score only those regions. Do not remove the recordings, infer
UEM from observed errors, or edit RTTM from these approximate listening notes.

The three recordings contribute 114.310 of the corpus's 218.273 false-alarm
seconds (52.37%) while providing only 28.226 scored reference speaker-seconds.
For scale only, recomputing the unchanged components on the remaining 97 files
gives DER 12.6176%, versus 15.1240% on all 100. This is a diagnostic sensitivity
calculation, not a corrected benchmark: removing recordings after observing
their errors would be cherry-picking.

Reproduce the focused audit:

```bash
python -m experiments.analyze_diarization_errors \
  --audio_dir .venv/viyt-data/audio \
  --ref_rttm_dir results/viyt_der_20261102/reference_rttm \
  --hyp_rttm_dir results/overclustering_backend_sweep_20261102/config_000/rttm \
  --output results/error_analysis/suspected_partial.json \
  --csv results/error_analysis/suspected_partial.csv \
  --recording_id C4HFERLpRfI --recording_id Uaq5N3FbQb4 \
  --recording_id ed0Vt0NxJUI --run_vad \
  --vad_threshold 0.5 --vad_min_silence_ms 200 --collar 0.25
```

## SpeakerError by speaker-count correctness

| Group | Recordings | Reference speaker-seconds | SpeakerError seconds | SpeakerError rate |
| --- | ---: | ---: | ---: | ---: |
| Correct predicted count | 60 | 2698.788 | 72.322 | 2.68% |
| Incorrect predicted count | 40 | 1730.249 | 326.930 | 18.89% |
| Overcount subset | 26 | 1061.024 | 132.389 | 12.48% |
| Undercount subset | 14 | 669.224 | 194.540 | 29.07% |

Most confusion is associated with an incorrect speaker count, especially
undercounting. The correct-count subset is not error-free: its largest temporal
assignment errors are `98EV6H0ZRlU` (15.312 seconds), `B5rLHPUJ8qc` (11.810),
`jxTaydBnwIg` (9.000), and `XSC94UbpoRQ` (7.250). These are the next recordings
to inspect for turn-boundary smoothing, mixed windows, or label switches.

Because the three manually reviewed files have unreliable one-speaker labels,
the count breakdown above is also recalculated without them as a sensitivity
check:

| Group (97-file diagnostic subset) | Recordings | Reference speaker-seconds | SpeakerError seconds | SpeakerError rate |
| --- | ---: | ---: | ---: | ---: |
| Correct predicted count | 60 | 2698.788 | 72.322 | 2.68% |
| Incorrect predicted count | 37 | 1702.023 | 326.765 | 19.20% |
| Overcount subset | 23 | 1032.799 | 132.224 | 12.80% |
| Undercount subset | 14 | 669.224 | 194.540 | 29.07% |

Count accuracy on this subset is 61.86%, with 23.71% overcounting and 14.43%
undercounting. The conclusion is unchanged: incorrect speaker count is strongly
associated with SpeakerError, and undercounted files have the highest error
rate. This exclusion only prevents known-bad count labels from contaminating
the diagnosis; it is not a replacement evaluation score.

Generate the corpus split without rerunning inference:

```bash
python -m experiments.analyze_diarization_errors \
  --audio_dir .venv/viyt-data/audio \
  --ref_rttm_dir results/viyt_der_20261102/reference_rttm \
  --hyp_rttm_dir results/overclustering_backend_sweep_20261102/config_000/rttm \
  --output results/error_analysis/all.json \
  --csv results/error_analysis/all.csv --collar 0.25
```

## Temporal diagnostics and visualization

Speaker-count status is independent from error observations. The status is one
of `correct_count`, `overcount`, or `undercount`; `observations` is a validated,
multi-valued reviewer field such as `speaker_assignment`, `boundary_error`,
`vad_miss`, `vad_false_alarm`, `possible_mixed_embedding`, `reference_issue`, or
`unknown`. DER components do not automatically establish any physical cause.

The analyzer uses the scorer-prepared reference, hypothesis, effective UEM,
collar, overlap policy, and global label mapping. Its JSON contains atomic and
merged intervals for correct assignments, speaker confusion, missed speech,
false alarms, long errors, and speaker-change offsets. Overlap is measured in
speaker-seconds, consistent with pyannote.

Render the four high-error correct-count recordings and all 14 undercounts from
an existing analysis without rerunning inference:

```powershell
python -m experiments.visualize_diarization_errors `
  --diagnostics results/error_analysis/all.json `
  --output_dir results/error_analysis/correct_count `
  --recording_id 98EV6H0ZRlU --recording_id B5rLHPUJ8qc `
  --recording_id jxTaydBnwIg --recording_id XSC94UbpoRQ

python -m experiments.visualize_diarization_errors `
  --diagnostics results/error_analysis/all.json `
  --output_dir results/error_analysis/undercount `
  --speaker_count_status undercount --format png
```

Pass `--uem data/official_uem` to the analyzer only for an externally supplied,
protocol-valid UEM. Excluded regions never appear as errors. Timeline figures
show reference lanes, globally mapped hypothesis lanes, component intervals,
recording ID, total temporal error, collar, and overlap configuration.

For annotation review, prepare a JSON list with `recording_id`, `audio_path`,
`reference`, `vad`, `suspected_unannotated`, and `speaker_changes`, then run:

```powershell
python -m experiments.visualize_diarization_errors `
  --review_manifest results/error_analysis/reference_review.json `
  --output_dir results/error_analysis/reference_review
```

The resulting waveform overlays are review evidence only. VAD regions and
approximate change markers must not be promoted to ground truth. A human should
listen, annotate independently, adjudicate disagreements, and preserve proposed
RTTM/UEM changes as separate versioned artifacts. The published reference must
remain unchanged until the benchmark owner confirms the intended coverage.

## Centroid merge diagnostics

Every attempted fragment now records all candidate target cosine similarities,
the best and second-best score, score margin, acceptance, and every rejection
reason. Each configuration also writes `centroid_merge_summary.json` with score
statistics and corpus-level reason counts.

The full VAD-200/ AHC-0.75 run produced the same evidence for similarity
thresholds 0.75, 0.80, and 0.85:

| Statistic | Value |
| --- | ---: |
| Merge attempts / accepted / rejected | 84 / 0 / 84 |
| Best-similarity minimum / maximum | -0.082 / 0.534 |
| Best-similarity mean / median | 0.293 / 0.312 |
| All target similarities | 207 |
| All-target minimum / maximum | -0.145 / 0.534 |
| All-target mean / median | 0.206 / 0.227 |
| Below similarity threshold | 84 attempts |
| Also below required margin | 17 attempts |

Every best candidate was below even the lowest configured threshold, so the
similarity gate—not fragment-size selection or an implementation failure—is the
primary reason no merge occurred. Lowering the threshold to approximately 0.5
would affect only a few candidates and is not justified without reference-safe
development evidence that those fragments belong to the same speaker. The
highest observed candidate was recording `OetAPu_5wDU` at 0.534.

With the embedding cache, the initial 100-recording extraction took 122.45
seconds. Each compatible centroid configuration reused all 100 cached entries
and completed in 1.74–1.83 seconds with unchanged DER 15.1240.

## AHC stability on development data

The new `ahc_stability` experiment contains only 0.70, 0.72, 0.74, 0.75, and
0.77. It is intentionally excluded from `--experiment all` so it is not run
accidentally on the reused test set.

```bash
python -m experiments.run_diarization_experiments \
  --audio_dir /path/to/development/audio \
  --ref_rttm_dir /path/to/development/reference_rttm \
  --model exps/exp1/model/model_0034.model \
  --output_dir results/dev_ahc_stability \
  --experiment ahc_stability --vad_min_silence_ms 200 \
  --window_sec 1.5 --device cuda --seed 42 --collar 0.25
```

Do not run this selection sweep on the old test set. After choosing one threshold
on development data, evaluate it once on a new, previously unused test set.

## Embedding cache

The experiment runner keeps an in-memory cache keyed by audio identity,
checkpoint/encoder identity, VAD parameters, window parameters, device, and
seed. AHC thresholds, clustering backend, known-count mode, and centroid
post-merge can reuse embeddings. Changing VAD or segmentation creates a cache
miss. `EmbeddingCacheHits` is written to the corpus results and each recording
records `embedding_cache_hit`.

A one-recording GPU smoke run measured 3.045 seconds for initial extraction and
0.013–0.016 seconds for each of three compatible centroid configurations. These
timings validate cache behavior only; they are not corpus quality measurements.

## Embedding-stage diagnostics

For stage-level attribution, use
[`EMBEDDING_ERROR_ANALYSIS.md`](EMBEDDING_ERROR_ANALYSIS.md). The companion
workflow preserves the production RTTM while exporting normalized ECAPA windows,
the exact AHC merge hierarchy, postprocessing labels, reference-overlap
classifications, and optional smoothing/short-merge counterfactuals. It also
defines the four plots and the required audio/checkpoint/configuration hashes.

At the time of this documentation update, the epoch-34 checkpoint is present but
the 19 target recording WAVs are unavailable. Consequently their embedding,
AHC, and postprocessing evidence remains `unknown`; the repository's synthetic
tests validate implementation behavior only. Do not interpret the historical
correct-count/undercount/overcount ledger as newly measured results.
