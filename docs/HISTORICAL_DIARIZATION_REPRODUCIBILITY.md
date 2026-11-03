# Historical diarization reproducibility audit

## Scope and conclusion

This audit compares the preserved November 2 ViYT-Diar experiments with the
November 3 embedding-analysis artifacts.  The observed five-recording speaker
count discrepancy is **explained**.  The exact historical dirty source remains
only partially reconstructable.

The preserved evidence confirms an architecture mismatch in the three ECAPA
`Bottle2neck` blocks:

- the historical multi-speaker run was recorded at `fc2eb0e16d6a4aab9263747c192985a42f1d137e`
  with a dirty worktree; the dirty change is strongly identified as
  `Conv1d -> ReLU -> BatchNorm1d` by the later commit `a78321de6c46e86161099e99f3170ea2032d4403`;
- the audited diagnostics HEAD `f2a05ea` does not contain `a78321d` and uses
  the original `Conv1d -> BatchNorm1d -> ReLU` expression;
- matched checkpoint-audit inference reproduces the near-collinear geometry
  only with `Conv -> BN -> ReLU`;
- after changing only that production expression to `Conv -> ReLU -> BN`, a
  controlled CUDA end-to-end run reproduced all five historical RTTMs exactly.

This is not a change to the top-level ECAPA input convolution: that layer is
`Conv -> ReLU -> BN` in both sources.  The mismatch is specifically
`Bottle2neck.forward()` and is applied in `layer1`, `layer2`, and `layer3`.

The exact historical dirty patch and historical embedding arrays were not
preserved.  Consequently, history cannot prove that this one line was the only
dirty change present in the old worktree.  It does establish that the line is
sufficient to explain the observed speaker counts and RTTMs.

## Preserved automatic-AHC comparison

The closest matched historical experiment is
`results/overclustering_backend_sweep_20261102/config_000`.  Its manifest and
the current per-recording configs agree on all recorded inference inputs:

| Field | Historical | Current |
|---|---|---|
| Checkpoint SHA256 | `58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325` | same |
| Method | AHC, automatic speaker count | same |
| Cosine-distance threshold | `0.75` | same |
| VAD threshold / minimum silence | `0.5` / `200 ms` | same |
| Window / step | `1.5 s` / `0.75 s` | same |
| Maximum speakers | `10` | same |
| Centroid merge | disabled | same |
| Device | CUDA | CUDA |
| Seed | `42` | `42` |
| Python / PyTorch | `3.14.3` / `2.11.0+cu130` | not recorded in current artifact |
| Git provenance | `fc2eb0e...`, dirty | not recorded in current artifact |

Every focus WAV still matches the SHA256 stored in the historical manifest.

| Recording | Audio SHA256 | Historical speakers | Current speakers | Current maximum AHC distance |
|---|---|---:|---:|---:|
| `98EV6H0ZRlU` | `39744cee5238dcb2e5ca5e0db8a47aaa1a789923a1e7bd4ee78da6329e828b1e` | 2 | 1 | 0.006754 |
| `B5rLHPUJ8qc` | `8d36a851862676b7bc27f32b2d4db41db5537fe952f10e88c77ae110d37b7623` | 2 | 1 | 0.009410 |
| `jxTaydBnwIg` | `64f524ce2049b04b5cbeaf555d6f55196f996e82a4b67913f4fbe6bf9a47f46e` | 2 | 1 | 0.004523 |
| `XSC94UbpoRQ` | `5afe625946edc7dd41c2465cc5e66693302713cec98f2e76f7aea40c9004650e` | 2 | 1 | 0.005173 |
| `jE9ZB-zqEgs` | `0b4c19eabaf4f68b79781d41bbe0f1c99b7825921bbc6a68135871e2c2d4cd7e` | 10 | 1 | 0.007678 |

The current distances are orders of magnitude below `0.75`, so the current
one-cluster result is the direct, expected result of its saved embeddings and
is not caused by temporal postprocessing.

## Controlled end-to-end reproduction

The Lead reran the production pipeline on CUDA after the single production
change in `Bottle2neck.forward()` from `Conv -> BN -> ReLU` to
`Conv -> ReLU -> BN`.  The run used the same checkpoint and focus WAVs, AHC
threshold `0.75`, VAD threshold `0.5`, minimum speech `250 ms`, minimum silence
`200 ms`, speech padding `30 ms`, window `1.5 s`, step `0.75 s`, minimum segment
`0.5 s`, seed `42`, maximum speakers `10`, and disabled centroid merge.

The output is preserved at
`results/ecapa_checkpoint_audit/corrected_smoke`.  It reproduced not only the
historical counts but the exact historical RTTM SHA256 for every focus case:

| Recording | Historical speakers | Corrected speakers | Historical and corrected RTTM SHA256 |
|---|---:|---:|---|
| `98EV6H0ZRlU` | 2 | 2 | `9dbb7d7535c8508e400517fff52728ac38f2d4699c0cafa552972ad514968884` |
| `B5rLHPUJ8qc` | 2 | 2 | `6b29e1035de6c2ba653113e758a19ab4243992c573656fa2adaff3248b45b105` |
| `jxTaydBnwIg` | 2 | 2 | `a1423385998a6438f7df6970f9c59c2329bbffe77939c9fde4e980e8a103ae3c` |
| `XSC94UbpoRQ` | 2 | 2 | `0472c8324346096d500cf6f02ed5761bd73610b57181329acefaf50af433d98c` |
| `jE9ZB-zqEgs` | 10 | 10 | `9eccb8206dd7becf92f46d68e4b2dc8f2922ff3f9a7c63945b296d097f3a421e` |

Because all other recorded inputs were held fixed and all five RTTM hashes
match, the `Bottle2neck` operation order is a confirmed causal explanation for
the observed historical/current contradiction.  The remaining uncertainty is
source provenance, not reproduction of this outcome.

The earlier `overclustering_ahc_sweep_20261102` run is also preserved.  At
thresholds `0.65`, `0.70`, `0.75`, and `0.80`, it produced respectively:

| Recording | 0.65 | 0.70 | 0.75 | 0.80 |
|---|---:|---:|---:|---:|
| `98EV6H0ZRlU` | 3 | 2 | 1 | 1 |
| `B5rLHPUJ8qc` | 2 | 2 | 2 | 2 |
| `jxTaydBnwIg` | 3 | 3 | 2 | 2 |
| `XSC94UbpoRQ` | 6 | 3 | 2 | 1 |
| `jE9ZB-zqEgs` | 10 | 10 | 10 | 7 |

That run was based on `86524ffbc2c448284e8e580f7e910f482f987ea7`
and was also dirty.  It demonstrates that the historical geometry was not the
near-zero-distance geometry preserved on the current branch.

## Git reconstruction

The repository began with `Conv -> BN -> ReLU` in commit
`a79763c2535949c3fc09e53abfcaae80da42eb41`.  The historical backend manifest
was written on November 2 at approximately 17:12 local time and records
`fc2eb0e...` plus `git_dirty=true`.  At 23:54, commit `a78321d` on
`skylak33/diarization-overclustering` changed exactly this expression to
`Conv -> ReLU -> BN` and added a checkpoint-compatibility test.

The current diagnostics branch is not a descendant of `a78321d`; both lines
have merge base `717c18659f78ae1dec21c9d6e688a55887f2826d`.  Thus the diagnostics branch
did not later revert the fix.  It was created from a lineage that never
received it.  In this precise sense the current branch regressed to the
checkpoint-incompatible block order.

No committed changes between the matched sources alter checkpoint loading,
audio decoding, VAD, cosine AHC distance, or normalization in a way that
explains the observed geometry.  Later `pipeline.py` and `cluster.py` changes
add caches and diagnostics or fix spectral clustering; the matched comparison
uses automatic AHC with centroid merge disabled.

## Independent corroboration

`results/ecapa_checkpoint_audit/checkpoint_health.json` performs matched
inference with the same checkpoint while changing only the `Bottle2neck`
operation order.  On 28 sampled real-audio embeddings:

| Geometry metric | `Conv -> BN -> ReLU` | `Conv -> ReLU -> BN` |
|---|---:|---:|
| Normalized centroid norm | 0.996842 | 0.309634 |
| Mean pairwise cosine | 0.993461 | 0.062387 |
| Minimum / maximum cosine | 0.976498 / 0.999507 | -0.326405 / 0.848615 |
| Effective rank | 5.64 | 17.52 |

The separately preserved VLSP comparison in
`results/vlsp_order_fix_20261102/comparison.json` used the same checkpoint and
20,000 fixed trials.  It reports EER `27.16%` for `Conv -> BN -> ReLU` and
`9.07%` for `Conv -> ReLU -> BN`.  This establishes that operation order is a
causal explanation for the current embedding pathology, not merely a Git
correlation.

## Evidence classification

### Confirmed

- Historical and current focus recordings use identical WAV hashes.
- Historical and current runs use the same checkpoint hash and matched AHC,
  VAD, window, step, seed, and device settings where recorded.
- Historical RTTMs contain `2, 2, 2, 2, 10` speakers; current RTTMs contain one.
- The audited HEAD `f2a05ea` and the saved current artifacts use
  `Conv -> BN -> ReLU` in `Bottle2neck`; commit `a78321d` uses
  `Conv -> ReLU -> BN`.
- Matched inference shows that `Conv -> BN -> ReLU` causes the near-collinear
  embedding geometry.
- The one-line production correction reproduces the five historical speaker
  counts and historical RTTM hashes exactly.

### Strongly supported

- The historical dirty run used the `Conv -> ReLU -> BN` edit later committed
  as `a78321d`: its timing, branch, commit diff, checkpoint-compatibility test,
  RTTMs, independent matched inference, and exact end-to-end reproduction all
  agree.

### Unresolved

- The complete dirty-worktree diff at the instant of the historical run.
- Element-wise comparison with historical embeddings; none were found.
- Current artifact Python/PyTorch and Git metadata; current configs did not
  record them.
- Whether additional uncommitted historical changes affected outcomes outside
  the five exactly reproduced focus cases.

## Source inventory

- Historical manifest:
  `C:/Users/mina/Documents/diarize-vietnamese-speech/results/overclustering_backend_sweep_20261102/experiment_manifest.json`
- Historical focus RTTMs:
  `.../overclustering_backend_sweep_20261102/config_000/rttm/*.rttm`
- Historical threshold sweep:
  `.../overclustering_ahc_sweep_20261102/experiment_manifest.json`
- Current configs, embeddings, traces, and AHC trees:
  `C:/Users/mina/Documents/diarize-vietnamese-speech-diagnostics/results/embedding_analysis/`
- Architecture and matched-inference audit:
  `.../results/ecapa_checkpoint_audit/checkpoint_health.json`
- Controlled corrected end-to-end artifacts:
  `.../results/ecapa_checkpoint_audit/corrected_smoke/`
- Independent VLSP order comparison:
  `C:/Users/mina/Documents/diarize-vietnamese-speech/results/vlsp_order_fix_20261102/comparison.json`
- Relevant commits: `a79763c`, `fc2eb0e`, `a78321d`, `248e46b`.

