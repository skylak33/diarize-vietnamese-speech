# Clustering improvements v2: exploratory diagnostics

## Decision

No clustering v2 option is promoted to production. Fixed-threshold AHC remains
the default, `adaptive_ahc` retains its legacy behavior, and
`adaptive_ahc_robust` is a separate opt-in experiment. There is no independent
development result: the comparison below reuses all 100 ViYT-Diar test
recordings that were already used in earlier analysis.

## Implementation boundary

Legacy adaptive AHC still evaluates the candidate gap inside its own Tukey
reference sample. Robust adaptive AHC removes the largest candidate gap before
computing `Q3 + 1.5 * IQR`, requires a unique largest gap, and otherwise falls
back to fixed-threshold AHC. Both modes fit a complete average-linkage cosine
hierarchy once and deterministically cut that same tree; there is no second AHC
fit.

The estimator trace records `selection_reason`, `eligible_gap_count`,
`best_gap`, `runner_up_gap`, `gap_margin`, `outlier_cutoff`, and
`fallback_n_speakers`. These fields expose weak or tied evidence without
turning uncertainty into an automatic policy change. Known-count and non-AHC
combinations remain invalid.

## Cached reused-test comparison

The run used the epoch-34 checkpoint SHA-256
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`,
1.5/0.75-second windows, VAD minimum silence 200 ms, AHC distance 0.75,
no centroid merge, seed 42, full-audio scoring, a 0.25-second collar, and
overlap included. The encoder was loaded once. The baseline populated the
embedding cache; every later configuration reported 100 cache hits.

| Automatic configuration | DER | Miss | FA | SpeakerError | Count accuracy | Count MAE | Wall time (s) | Clustering (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed AHC 0.75 | 15.123975 | 1.181326 | 4.928228 | 9.014420 | 0.60 | 0.65 | 112.083978 | 0.323393 |
| Legacy adaptive AHC | 13.769524 | 1.181326 | 4.928228 | 7.659970 | 0.68 | 0.55 | 6.429086 | 0.299250 |
| Robust adaptive AHC | 13.013353 | 1.181326 | 4.928228 | 6.903799 | 0.71 | 0.49 | 6.357842 | 0.298129 |
| Spectral | 16.648815 | 1.181326 | 4.928228 | 10.539260 | 0.65 | 0.76 | 52.935768 | 46.197747 |
| NME-SC | 19.956147 | 1.181326 | 4.928228 | 13.846593 | 0.46 | 1.03 | 13.096978 | 6.590119 |

Wall times after the first row are cache-reuse timings and cannot be compared
with the extraction-inclusive 112.08-second baseline as standalone end-to-end
latency. Clustering time is the comparable backend-specific measurement. The
fixed baseline produced 100 RTTMs and had **0 SHA-256 mismatches** against the
frozen corrected-encoder baseline, so the v2 implementation did not silently
change the default output.

Against fixed AHC, legacy adaptive improved 15 recordings, worsened 11 and left
74 unchanged; robust adaptive was 20/13/67; Spectral was 33/17/50; and NME-SC
was 32/34/34. Robust versus legacy adaptive was 5/2/93. These are paired DER
directions with a `1e-9` equality tolerance, not significance tests.

This is a single paired exploratory run, not a parameter-selection study or
held-out benchmark. The lower pooled DER for robust adaptive AHC is therefore a
hypothesis for future independent development data, not a production claim.

## Per-recording behavior

The pooled gain hides material regressions. The five largest robust-versus-fixed
DER increases were:

| Recording | Reference K | Fixed K | Robust K | Fixed DER | Robust DER | Delta DER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Rr7GNFMEY-M` | 2 | 4 | 2 | 9.603869 | 50.637748 | +41.033880 |
| `sKPko99dPQ8` | 2 | 4 | 1 | 20.674160 | 38.025585 | +17.351425 |
| `BJJIC04U0ao` | 2 | 3 | 5 | 3.546924 | 12.311638 | +8.764714 |
| `8nkbJvn18nA` | 2 | 4 | 2 | 17.028932 | 24.452976 | +7.424044 |
| `blwCA5aG1s0` | 2 | 2 | 3 | 4.926909 | 10.963725 | +6.036816 |

Correct count is not sufficient for a good partition: `Rr7GNFMEY-M` moved from
four labels to the reference count of two while speaker-error time rose from
2.138699 to 22.769999 seconds. Conversely, the largest improvements were
`9lm02QThHjo` (DER 49.808780 to 0.927812), `IZxOT44FKsU` (48.673549 to
0.890403), and `HTwksOmzTsc` (39.641563 to 4.073033). These opposing tails are
why corpus averages alone do not justify promotion.

The hierarchy evidence is strongest for `IZxOT44FKsU`: robust adaptive found a
best merge gap of `0.181845` versus runner-up `0.043293` and changed the count
from one to two. For `9lm02QThHjo`, the margin was much smaller (`0.080403`
versus `0.061303`); robust AHC initially produced three clusters and smoothing
reduced them to the correct count of two. Both outcomes agree with their high
diagnostic pairwise AUCs (`0.983` and `0.931`), but reference labels were used
only after inference to interpret the result.

`jE9ZB-zqEgs` illustrates persistent fragmentation. Fixed AHC produced ten
clusters and DER `69.745598`; both adaptive variants produced eight and DER
`60.113566`. Its diagnostic AUC is only about `0.63`, with substantial
same/different-speaker similarity overlap, so a hierarchy cut alone does not
recover the two-speaker reference partition.

`V-cuzXOX9sM` remains a critical baseline failure: fixed AHC initially produced
six clusters for two reference speakers, smoothing reduced that to five, and
speaker-error time was 16.377930 seconds. Robust adaptive produced two labels,
but DER remained `56.441281` and SpeakerError `44.354597%`, again showing that
correct count is not correct assignment. Its diagnostic labels the evidence as
low-confidence AHC fragmentation, with mixed and uncertain windows also
plausible; it does not establish a single root cause.

## Stage-count analysis

The fixed-AHC baseline had 52 recordings with an incorrect initial cluster
count. Temporal smoothing decreased the count in 24 recordings and removed 31
cluster labels in total; centroid merge and final postprocessing changed no
counts in this run. Comparing absolute count error before and after all
postprocessing, 23 recordings moved closer to the reference count, none moved
farther away, and 77 were unchanged. Twelve of the 52 initial errors were
corrected exactly, leaving 40 final count errors and introducing none.

This does not show that smoothing is universally beneficial. For
`XJG-o9BK4To`, smoothing changed three initial clusters to one final identity
against a two-speaker reference. Its absolute count error stayed at one, but the
failure changed from overcounting to undercounting. The previously measured
no-smoothing counterfactual was also slightly worse in DER, so this tradeoff
does not support disabling or strengthening smoothing globally.

Thirty-four recordings had the correct final count but non-zero SpeakerError.
Speaker count is therefore a diagnostic dimension, not a substitute for DER or
assignment-quality inspection.

## Reproduction (PowerShell)

Use a new output directory. This command deliberately reproduces reused-test
analysis; it does not create held-out evidence:

```powershell
$audio = "data\viyt\audio"
$references = "data\viyt\reference_rttm"
$model = "exps\exp1\model\model_0034.model"
$output = "results\clustering_v2_reproduction"
$diagnostics = "results\clustering_v2_reproduction_diagnostics"

python -m experiments.run_diarization_experiments `
  --audio_dir $audio --ref_rttm_dir $references --model $model `
  --output_dir $output --diagnostic_output_dir $diagnostics `
  --experiment clustering_count --threshold 0.75 `
  --vad_min_silence_ms 200 --window_sec 1.5 --step_sec 0.75 `
  --device cuda --seed 42 --collar 0.25
```

Verify that the fixed configuration still matches a frozen baseline directory:

```powershell
$current = "results\clustering_v2_reproduction\config_000\rttm"
$frozen = "results\viyt_corrected_20261103\hypothesis_rttm"
$mismatches = foreach ($file in Get-ChildItem $current -Filter *.rttm) {
  $other = Join-Path $frozen $file.Name
  if (!(Test-Path $other) -or
      (Get-FileHash $file.FullName -Algorithm SHA256).Hash -ne
      (Get-FileHash $other -Algorithm SHA256).Hash) {
    $file.Name
  }
}
@($mismatches).Count
```

Regenerate the fixed-baseline stage report from saved diagnostics without
rerunning audio inference:

```powershell
python -m experiments.analyze_stage_counts `
  --artifacts "results\clustering_v2_reproduction_diagnostics\config_000" `
  --reference-rttm $references `
  --evaluation "$output\config_000\per_file.csv" `
  --output-dir "results\clustering_v2_reproduction_stage_analysis" `
  --expected-recordings 100
```

The generated `results/` artifacts are ignored by Git. Preserve their manifests,
CSV/JSON outputs and hashes separately when using them as evidence.
