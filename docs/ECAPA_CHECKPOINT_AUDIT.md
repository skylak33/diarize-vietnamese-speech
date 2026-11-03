# ECAPA checkpoint compatibility audit

## Result and scope

Checkpoint `exps/exp1/model/model_0034.model` has SHA-256
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.
Its 234 encoder tensors load strictly, the production loader preserves their
values, the encoder is placed in evaluation mode, and the observed outputs are
finite and 192-dimensional.

Those checks were necessary but insufficient. The checkpoint was trained with
the first operation in each `Bottle2neck` block ordered as:

```text
Conv1d -> ReLU -> BatchNorm1d
```

The audited inference source instead used:

```text
Conv1d -> BatchNorm1d -> ReLU
```

The tensor keys and shapes are identical, so `load_state_dict(..., strict=True)`
cannot detect the difference. Historical commit
`a78321de6c46e86161099e99f3170ea2032d4403` records the checkpoint-compatible
order. Production inference now uses that order, and a forward-hook regression
test verifies that each Res2Net block feeds rectified convolution output into
its first BatchNorm.

## Causal chain

1. Training accumulated BatchNorm running statistics after the ReLU.
2. Wrong-order inference applied those statistics before the ReLU.
3. The semantic mismatch propagated through all three residual Res2Net blocks.
4. On 1,317 saved ViYT-Diar windows, the resulting normalized embeddings had a
   centroid norm of `0.998002` and mean cross-recording similarity of about
   `0.995939`.
5. Their AHC merge distances (`0.002321`-`0.013472`) were far below the configured
   cosine-distance threshold `0.75`, so every audited recording merged to one
   cluster before temporal postprocessing.

This chain explains the observed near-collinearity and one-cluster behavior of
the saved diagnostic run. It does not establish the diarization quality of the
corrected encoder.

## Matched architecture experiment

`experiments/audit_ecapa_checkpoint.py` loaded the same checkpoint into both
operation orders and evaluated the same sampled audio. The audit included 12
ViYT-Diar windows and 16 audio sides from sampled VLSP trial pairs. These small
samples diagnose graph compatibility; they are not a verification benchmark.

| Measurement | Wrong order | Checkpoint-compatible order |
| --- | ---: | ---: |
| Normalized centroid norm | 0.996842 | 0.309634 |
| Effective rank | 5.640 | 17.524 |
| Mean pairwise cosine | 0.993461 | 0.062387 |

Mean matched cosine between embeddings from the two orders was `0.087566`.
The wrong-order live embeddings exactly reproduced the sampled saved embeddings
(maximum absolute error `0`), and repeated production extraction was deterministic
(maximum absolute error `0`). Twenty sampled WAV files had distinct SHA-256
hashes, and `utils.audio.decode_audio` exactly matched the VAD loading path.

The order comparison locates the fault before pooling rather than in L2
normalization. In the wrong-order sample, mean pairwise cosine was already near
one at frame and pooling stages. With the compatible order, the large common
direction remained before `bn5`, while `bn5` reduced sampled mean pairwise cosine
to approximately `0.06`. This is evidence about the sampled activations, not a
claim about speaker accuracy.

## Statistical audit

The old overlap estimator used at most 50 bins across the fixed cosine range
`[-1, 1]`. It could therefore put narrow distributions near one in the same
coarse bin. The corrected estimator uses common adaptive bounds and shared
equal-width Freedman-Diaconis bins capped at 50. On the 19 saved wrong-order
recordings:

| Metric | Result |
| --- | ---: |
| Old fixed-range overlap | 1.0 for all recordings |
| Adaptive overlap | 0.553817-0.914706; median 0.837309 |
| KS statistic | 0.042653-0.429695; median 0.151755 |
| Wasserstein distance | 0.000066-0.001377; median 0.000305 |
| Diagnostic AUC | 0.470921-0.734056; median 0.583723 |

The tie-aware AUC definition did not change. All pairwise results are descriptive:
overlapping windows and repeated comparisons are dependent, so no naive pairwise
p-values or independent-sample confidence claims are made.

## Exploratory mean-centering

As a geometry diagnostic only, the audit estimated one centroid from all 1,317
saved wrong-order ViYT-Diar embeddings, subtracted it, and renormalized the
residuals. This uses the reused test artifacts on both sides of the comparison;
it is not an unbiased evaluation and is not part of production inference.

| Measurement | Original | Test-estimated centered |
| --- | ---: | ---: |
| Normalized centroid norm | 0.998002 | 0.052982 |
| Effective rank | 20.574 | 27.662 |
| Mean pairwise cosine | 0.996004 | 0.002049 |
| Median per-recording diagnostic AUC | 0.583723 | 0.592813 |
| Median AHC speakers at unchanged threshold 0.75 | 1 | 4 |

Centering removes the dominant common direction and exposes residual structure,
but the modest median AUC change and test-estimated transform do not justify a
production normalization change. The architecture correction supersedes this
wrong-order diagnostic; any centering hypothesis requires a separate development
set and newly generated checkpoint-compatible embeddings.

## Historical reproducibility boundary

The operation-order regression is a concrete committed-code explanation for why
the current diagnostic branch disagreed with a checkpoint-compatible historical
source. It is also consistent with earlier runs reporting multiple clusters.
The exact old runs were based on an unavailable Git-dirty snapshot, so their
complete source and cached embeddings cannot be verified element by element.
However, a controlled current-source smoke run changed only the incompatible
operation order and reproduced all five focus counts (`2, 2, 2, 2, 10`) with the
recorded checkpoint and frozen inference configuration. All five corrected RTTM
SHA-256 hashes also match the preserved historical RTTMs. This explains the
observed contradiction while leaving the complete historical dirty source
snapshot unresolved.

## Artifact invalidation

All embeddings, caches, similarity summaries, AHC traces, RTTMs, DER, and
SpeakerError values derived from the wrong-order graph are invalid for evaluating
the corrected encoder. Keep them only as fault-reproduction evidence. Do not mix
old and regenerated embeddings in one corpus report, and do not relabel the old
results as a corrected baseline.

No AHC, VAD, centering, PLDA, or checkpoint parameters were changed. No threshold
was tuned on ViYT-Diar. The next justified action is to regenerate embeddings with
the fixed encoder, record source/checkpoint/audio hashes, and evaluate the frozen
configuration. Any later calibration must use independent development data.

## Reproduction on Windows PowerShell

Run the compatibility and statistical tests:

```powershell
python -m pytest -q tests/test_ecapa_compatibility.py `
  tests/test_ecapa_checkpoint_audit.py `
  tests/test_embedding_diagnostics.py
```

Run the checkpoint audit without VLSP data:

```powershell
python experiments/audit_ecapa_checkpoint.py `
  --checkpoint exps/exp1/model/model_0034.model `
  --artifacts results/embedding_analysis `
  --vity-audio C:\path\to\viyt-data\audio `
  --device cpu `
  --output results/ecapa_checkpoint_audit
```

`--vlsp-trials` and `--vlsp-audio` are optional and may both be omitted. If
independent VLSP data is available locally, provide both paths; the audit does
not download data.

Recompute descriptive statistics from saved artifacts without GPU inference:

```powershell
python experiments/audit_embedding_statistics.py `
  --artifacts results/embedding_analysis `
  --reference_rttm C:\path\to\reference.rttm `
  --output_dir results/ecapa_checkpoint_audit
```

Generated JSON and CSV files under `results/ecapa_checkpoint_audit` are analysis
artifacts, not source-controlled quality claims.
