# Confidence-aware speaker cluster merging

## Hypothesis and scope

Conservative AHC can preserve speaker-pure fragments while selecting too many
speakers. COCAM tests whether distribution-aware merging can reduce fragmentation
without introducing cross-speaker confusion. It is experimental and disabled by
default. Correct speaker count alone is insufficient evidence of a good partition.

The starting revision is `0ea114f` on
`skylak33/partition-quality-stability-analysis`; implementation lives on
`skylak33/confidence-aware-cluster-merging`. The remote starting branch had been
removed, but its exact local history was preserved. `main` was not modified.

## Rule

Initial labels come from the existing average-linkage cosine-distance Fixed AHC.
The threshold remains independently configurable, including `0.65` and `0.75`.
A smaller distance threshold retains more clusters. It is not a cosine-similarity
threshold: distance `0.75` corresponds to average similarity `0.25`.

Normalize input embeddings. For each proposed pair, compute:

- Centroid cosine similarity using all members.
- Median, q20, q80 and interquartile range of cross-member cosine similarities.
- Median off-diagonal within-cluster similarity, and the same statistic after
  hypothetically joining the bounded samples.
- Compactness drop: `min(compactness_a, compactness_b) - merged_compactness`.
- Symmetric margin between the pair's cross median and each member's strongest
  alternative pair. Both clusters must select one another as their best neighbor.

With `s_ij = median(cross similarities)`, the margin is

```text
m_ij = min(s_ij - max(s_ik for k != i,j),
           s_ij - max(s_jk for k != i,j))
```

If only two clusters remain, there is no alternative and the margin condition is
vacuous; all absolute evidence gates still apply. Equal alternative scores fail
the positive margin gate. Partner ties use the lower integer label.

Acceptance is a conjunction: sufficient support, valid centroids, centroid
threshold, median threshold, q20 threshold, compactness-drop threshold, mutual
nearest neighbors and symmetric margin. There is no weighted score or calibrated
probability. IQR and q80 are descriptive evidence, not additional tunable gates.
Reference speaker labels, DER, ARI and oracle speaker count never enter inference.

| Parameter | Strict default | Moderate development candidate |
|---|---:|---:|
| Initial AHC distance | 0.75 | 0.75 |
| Minimum centroid similarity | 0.65 | 0.55 |
| Minimum cross median | 0.25 | 0.20 |
| Minimum cross q20 | 0.10 | 0.05 |
| Maximum compactness drop | 0.12 | 0.12 |
| Minimum symmetric margin | 0.02 | 0.02 |
| Maximum sampled members per cluster | 32 | 32 |
| Minimum members per cluster | 2 | 2 |

These are uncalibrated heuristic operating points. The strict cross median is
the similarity corresponding to the initial distance cut. Positive q20 and
normalized-centroid evidence test consistency despite diffuse cluster members.
The moderate candidate reduces these absolute gates by small fixed amounts.
An earlier draft's q20 >= 0.65 was removed before examining any candidate results:
with all members sampled, it implies mean cross similarity >= 0.32 even allowing
the bottom fifth to equal -1, incompatible with two surviving AHC 0.75 clusters.
Neither final operating point was selected using ViYT-Diar or DataHive.

## Procedure and safeguards

```text
labels = existing_fixed_ahc(embeddings, distance_threshold)
while at least two clusters remain:
    compute missing pair evidence; reuse unaffected cached evidence
    recompute nearest neighbors and symmetric margins
    record each proposal and its failed gates
    if no acceptable pair remains: stop
    select maximum (q20, median, centroid similarity, -label_a, -label_b)
    join the selected pair; invalidate affected evidence
return labels and optional diagnostics
```

An accepted merge reduces K exactly once, so at most `initial_K - 1` merges occur.
Cluster labels are retained as integers. Empty input is a no-op; singleton
clusters abstain. Nonfinite/zero embeddings, malformed timestamps, noninteger
labels and invalid configuration are rejected. Cancelled centroids abstain.
Overlapping embedding windows are unioned when reporting cluster duration.
Duration and size are descriptive support, never sufficient merge evidence.

Samples use evenly spaced temporal indices. Each cross matrix is at most
`32 x 32`; hypothetical compactness uses at most 64 vectors. Pair evidence is
recomputed only after a member cluster changes. Pair storage is O(K²), robust
pair calculations are O(m²d), and all-round margin bookkeeping is at most
O(K³ log K). Saving every round's proposals can use O(K³) diagnostic records.
Large K remains a limitation; bounded sampling prevents unbounded within-pair
quadratic memory, not all clustering costs. Initial AHC retains its existing cost.

Unlike the existing centroid merger, COCAM can join two substantial clusters and
requires distribution, compactness and symmetric-alternative evidence. The old
merger considers small fragments (at most four windows by default), strictly
larger targets, centroid threshold `0.80` and margin `0.05`. Its code is unchanged.

Controlled tests include matching centroids with incompatible distributions,
outliers, singleton bridges, unequal sizes, unsafe chains, cancelled centroids,
and deterministic diagnostics. A symmetric three-fragment single-speaker example
abstains because alternative evidence is ambiguous. This is an explicit
fragmentation limitation, not successful speaker recovery.

## Integration

```yaml
clustering:
  method: ahc
  count_estimator: backend_default
  threshold: 0.75
postprocessing:
  centroid_merge:
    enabled: false
  confidence_aware_merge:
    enabled: true
```

The strict defaults above are used unless overridden in YAML. The pipeline order
is Fixed AHC, optional confidence merging, smoothing (window 3), overlap
partitioning, short-segment merging (0.3 seconds), consecutive merging and RTTM.
The two merger flags are mutually exclusive. Confidence merging rejects adaptive
estimators, other clustering backends and explicit speaker counts.

`pipeline_trace.json` preserves initial, after-centroid, after-confidence,
after-smoothing and final partitions. `confidence_merge_decisions.json` records
accepted/rejected proposals, evidence, failed gates and cluster counts. Existing
counterfactual analysis uses the post-confidence partition when selected.

## Experimental protocol

The clean synthetic development corpus contains 300 conversations, 60 each with
1–5 speakers. Its 323 source speaker IDs are disjoint from validation and holdout;
the audit also checks source recording IDs and content hashes. Audio, metadata
and RTTM hashes are compared against the historical benchmark manifest.
Speaker identity equivalence across source datasets and overlap with ECAPA
training identities cannot be established from these identifiers alone.

Historical synthetic artifacts did not contain saved embeddings. One CPU
extraction pass therefore creates a resumable archive per development recording,
using the frozen checkpoint, unchanged mono/VAD/windowing code, and seed 42.
Every method receives exactly the same cached embeddings. The experiment uses
full-pipeline speech regions, VAD silence 100 ms, window 1.5 s, step 0.75 s,
minimum segment 0.5 s, collar 0.25 s and scored reference overlap.

All eight required baselines/candidate families are evaluated; the confidence
family has only the two predetermined operating points above. The selection rule
was committed before examining results: eligibility requires pooled DER no more
than 0.25 percentage points worse than Fixed 0.75, purity loss at most 0.005, and
at most 1% of recordings regressing by more than 10 DER points. Eligible candidates
are ranked by pooled DER, SpeakerError, count MAE, ARI, purity, fragmentation,
count accuracy and method name. If neither qualifies, the lower-DER candidate is
retained only for descriptive diagnostics, with no deployment recommendation.

Metrics are recorded at initial, merged, smoothed and final stages. Before final
RTTM, overlapping windows are partitioned into a single label per time point.
ARI, purity and mean reference-speaker fragmentation use the same unambiguous
embedding-window sample at every stage; final labels are projected to windows
by maximum temporal overlap. Unlabeled/ambiguous windows are excluded from those
partition metrics but remain in official DER. Corpus DER/SpeakerError pool scored
speaker seconds. Partition summaries weight by eligible window count. Final K
is measured on the complete output, including windows excluded from ARI.

Stability uses the first 25 development conversations, balanced five per K:
angular rotation 0.02 radians and random 75% window retention, seeds 11 and 23.
It records output K changes, ARI against the common baseline windows, accepted
merge-count changes, and changed merge links. A merge link is a window pair in
different initial clusters but the same post-merge cluster; link changes are
label-permutation invariant. Stability is not an inference-time gate.

Reference-dependent merge audits are computed after inference, replaying accepted
merges to measure ARI/purity and dominant-reference identity crossings. Dominant
identity crossings do not capture every mixture involving already impure clusters.
Paired differences are candidate minus baseline, with improved/worsened/unchanged
counts and worst cases. No confidence interval treats synthetic conversations
as independent real conversations: speakers and source utterances recur.

## Reproduction

Set `$history` to the local historical results root; no command downloads data.

```powershell
$history = 'C:\Users\mina\Documents\diarize-vietnamese-speech-diagnostics\results'
$output = 'results/confidence_aware_merging'
python experiments/benchmark_confidence_aware_merging.py prepare `
  --conversations "$history/synthetic_vietnamese_benchmark/generated" `
  --historical-manifest "$history/synthetic_vietnamese_benchmark/experiments/development/manifest.json" `
  --output "$output/cache_development"
python experiments/benchmark_confidence_aware_merging.py evaluate `
  --cache "$output/cache_development/manifest.json" `
  --output "$output/development" --stability-recordings 25
```

`prepare` resumes only compatible, hash-verified archives. Evaluation refuses an
existing output protocol to avoid overwriting experiments. Use a fresh output
directory for reproduction. The candidate and source hashes are saved in
`development/frozen_candidate.json` before exploratory evaluation.

Use `import-artifacts --artifacts <directory> --references <RTTM directory>
--output <cache manifest>` to index existing DataHive/ViYT diagnostic archives.
Then run `evaluate --cache <manifest> --frozen
<development/frozen_candidate.json> --output <fresh exploratory directory>`.
These comparisons use identical cached embeddings within each corpus. ViYT's
historical frontend uses silence 200 ms; DataHive uses 100 ms. Frontends are not
changed to make cross-corpus results appear comparable.

## Measured results (2026-11-08)

The candidate was frozen at `f76c7e4`, before exploratory evaluation. Selected
configuration: `confidence_moderate`, also provided in
`configs/diarization_confidence_aware_experimental.yaml`. Algorithm SHA256:
`241132be61056e398dec0e85b774c3ade9fa47260350cfe945a6f286f445cf13`.
The default experimental gates remain the strict operating point; selecting the
YAML explicitly reproduces the development-selected moderate point.

Development, 300 clean synthetic conversations, full pipeline, final partition:

| Method | DER % | SpeakerError % | Count accuracy % | ARI | Purity | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|
| Fixed AHC 0.65 | 9.2713 | 8.8354 | 48.33 | 0.8546 | 0.9952 | 1.3998 |
| Fixed AHC 0.75 | 6.0574 | 5.6215 | 69.33 | 0.9117 | 0.9772 | 1.1410 |
| Fixed 0.75 + existing centroid | 6.0574 | 5.6215 | 69.33 | 0.9117 | 0.9772 | 1.1410 |
| Legacy Adaptive | 9.1993 | 8.7635 | 57.67 | 0.8127 | 0.9668 | 1.2649 |
| Robust Adaptive | 10.6547 | 10.2188 | 55.00 | 0.7761 | 0.9612 | 1.3557 |
| Spectral | 8.8421 | 8.4062 | 63.67 | 0.9006 | 0.9969 | 1.4124 |
| NME-SC | 17.9660 | 17.5301 | 63.00 | 0.7915 | 0.9957 | 1.7899 |
| COCAM strict | 6.0574 | 5.6215 | 69.33 | 0.9117 | 0.9772 | 1.1410 |
| COCAM selected moderate | 5.8953 | 5.4594 | 69.67 | 0.9150 | 0.9772 | 1.1378 |

Count MAE decreases from 0.3667 to 0.3633. Only `development_000291` improves;
299 recordings are unchanged and none has worse DER. Paired mean DER change is
-0.1596 points, median zero; pooled change is -0.1621 points. This is one
successful example, not broad evidence of better clustering.

| Reference K (60 conversations each) | Fixed DER % | COCAM DER % | Fixed count accuracy % | COCAM count accuracy % |
|---|---:|---:|---:|---:|
| 1 | 5.8362 | 4.9461 | 83.33 | 85.00 |
| 2 | 4.8076 | 4.8076 | 73.33 | 73.33 |
| 3 | 3.6874 | 3.6874 | 71.67 | 71.67 |
| 4 | 6.1570 | 6.1570 | 66.67 | 66.67 |
| 5 | 8.8781 | 8.8781 | 51.67 | 51.67 |

The sole accepted development merge joins two 19-window fragments of one speaker:
centroid 0.5595, cross median 0.2289, q20 0.1418, compactness drop 0.0655.
K changes 2 to 1, ARI 0 to 1, purity stays 1, fragmentation 2 to 1, and DER
48.9091% to 1.0386%. Smoothing retains this improvement. The centroid baseline
cannot join these equally sized substantial clusters under its existing rule.
Moderate COCAM accepts 1 of 1,439 proposals; the other 1,438 abstain. Primary
failed gates are centroid similarity (1,393) and insufficient support (45).
These counts include reexamined pairs across rounds; full failed-gate lists
remain available. Strict COCAM accepts none.

### Frozen exploratory comparisons

These are exploratory reused-test evaluations, not independent validation.

| Corpus | Fixed/centroid DER % | COCAM DER % | Fixed SpeakerError % | COCAM SpeakerError % | Improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|
| DataHive, 8 recordings | 13.9945 | 13.9383 | 8.1947 | 8.1385 | 1 / 0 / 7 |
| ViYT-Diar, 100 recordings | 15.1240 | 14.9281 | 9.0144 | 8.8186 | 2 / 0 / 98 |

DataHive accepts 1/361 proposals (`vi_000003`, initial K 10 to 9); 360 reject.
Its primary rejected gates are centroid (264), support (78), median (9), mutual
neighbor (5), margin (4). ViYT accepts 2/367 proposals; 365 reject, primarily
centroid (253) and support (112). One ViYT merge crosses dominant reference
speaker identities. Thus only three of four accepted merges across all three
corpora have matching dominant reference identities.

DataHive final count accuracy remains 0/8, MAE 4.375 to 4.125. ViYT count accuracy
remains 60%, MAE 0.65 to 0.63. DataHive pooled DER change is -0.0562 points
(paired mean -0.0887, median zero); ViYT pooled change is -0.1959 points
(paired mean -0.2015, median zero). Historical ViYT DER was 15.123975%; the
present 15.123985% baseline scores full-precision geometry before textual RTTM
millisecond rounding. SpeakerError reproduces 9.014420%.

There are no final DER regressions, but there is an important partition failure:
`V-cuzXOX9sM` merges different speakers. Immediate purity falls 0.90625 to
0.78125 and ARI 0.39836 to 0.22223. After smoothing its purity is 0.71875,
versus baseline 0.875, and ARI 0.10821 versus 0.30783. Despite that, final DER
improves 63.3099% to 52.7956% under global speaker matching. Lower DER therefore
does not establish speaker-purity preservation. No observed DER regression is
not evidence that the acceptance criterion prevents destructive identity merges.

Postprocessing also matters in DataHive. Corpus COCAM DER changes 12.4954%
(initial), 12.4111% (merged), 13.9383% (smoothed/final). ARI changes 0.83034,
0.83056, 0.72811; purity 0.98273, 0.98273, 0.94456. Smoothing consumes part of
the merge benefit. Historical DataHive purity 0.9870 was a duration-weighted
initial-partition metric; it should not be compared directly with this table's
equal-window final-partition purity. `vi_000004` receives no new merge and its
known hierarchy failure remains unresolved.

### Stability and efficiency

On the balanced 25-recording development probe, Fixed/centroid/COCAM have no K
changes under angular noise (50 trials each, mean ARI 1). With 75% subsampling,
each changes K in 6/50 trials, mean ARI 0.9289. Robust changes K in 3/50 angular
trials (ARI 0.9922) and 13/50 subsampling trials (ARI 0.8733). However, no COCAM
merge is accepted in that subset, so its apparent merge stability is vacuous.

A separate post-hoc probe targets all four recordings with accepted merges.
COCAM changes K in 4/8 angular trials (mean ARI 0.8883) and 6/8 subsampling
trials (ARI 0.7159), compared with Fixed's 0/8 (ARI 0.9507) and 2/8 (ARI 0.6141).
Changed merge links total 10,480 and 6,202 respectively; these are correlated
window-pair counts, not independent decisions. The DataHive and `jE9ZB-zqEgs`
merges disappear under both angular seeds. The incorrect `V-cuzXOX9sM` merge
persists under both angular seeds: stable decisions can still be wrong.
No parameters were changed after these observations.

CPU clustering/postprocessing totals over the 300 dev recordings are 0.281 s
Fixed, 0.290 s centroid and 0.928 s COCAM, excluding cached frontend extraction
and evaluation. Single-input profiles with allocation tracing: 46-window/K5
input, 0.0032 s/0.114 MiB Fixed versus 0.0139 s/0.220 MiB COCAM; 1,342-window/K10
DataHive input, 0.1997 s/9.83 MiB Fixed versus 0.2946 s/9.83 MiB COCAM.
Traced allocations are not total process RSS; AHC dominates the larger peak.
These single-run timings establish practical scale, not statistically precise
performance ratios.

Artifacts are under `results/confidence_aware_merging/`: split audit, resumable
cache, development/frozen candidate, per-stage results, proposal traces,
reference merge audits, paired comparisons, balanced stability, exploratory
corpora, and `posthoc_analysis/` profiles/accepted-case stability. Reproduce the
last analysis with:

```powershell
python experiments/analyze_confidence_merge_results.py --root results/confidence_aware_merging
```

The analysis also refuses an existing output directory. Artifacts, audio,
embeddings and RTTM collections are excluded from Git. Recommendation: retain
COCAM as experimental. Its measured dev gain is concentrated in one single-speaker
conversation, K=2–5 do not improve, accepted merges are unstable in targeted
checks, and one reused-test merge damages speaker purity. Independent validation
remains necessary before replacing Fixed AHC.

Final verification: `python -m pytest -q` reports 348 passed, 0 failed, 0 skipped;
`python -m compileall -q diarize eval experiments tests` and `git diff --check`
succeed. The experimental YAML exactly matches the frozen effective parameters.

## Scientific limitations

Synthetic conversations reuse source voices and utterances, contain clean
non-overlapping concatenated speech, and do not represent hundreds of independent
natural dialogues. The existing validation/holdout splits were already examined.
No verifiably untouched local diarization pool with established source/training
separation is available for an independent claim. Development and reused-test
results therefore do not establish production generalization.

DataHive has eight nominally two-speaker recordings and channel-VAD silver
references. Targeted listening confirmed overlap in 23 excerpts and found a
channel-label error in one; the full corpus is not human-verified. Historical
references remain untouched. ViYT-Diar is also repeatedly inspected evidence.

The rule can reject genuine acoustic changes, leave singleton speakers
fragmented, or accept two different but acoustically similar voices. Sampling
can miss minority modes and depends on temporal ordering; overlapping windows
are correlated support. Median compactness can hide a small incompatible mode.
Initial AHC errors cannot be split or repaired by any subsequent merge-only rule.
Positive margins also inhibit merging three similarly spaced fragments of one
speaker. Stability and purity do not by themselves establish correct identities.

The checkpoint SHA256 is
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`;
encoder SHA256 is
`96cd2147b769f3730d11eaabefedb3f3b52c3913f87f74d01b208d4bf92b14d8`.
AST comparison against the starting revision confirmed all existing clustering,
smoothing and frontend functions are unchanged. ECAPA, VAD and loader files are
unchanged as well. Further work should prioritize a truly independent natural
development/validation corpus and better merge evidence before more heuristics.
