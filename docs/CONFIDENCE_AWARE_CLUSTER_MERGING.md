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
