# Speaker merge decision separability

## Conclusion

Existing similarity features rank pure same-speaker pairs meaningfully, but the
current rule has almost no recall. Its centroid gate is the binding constraint.
Feature overlap and mixed initial clusters prevent interpreting a more permissive
development-only operating region as a safe deployment rule. These observations
support further investigation, not COCAM V2 or a production threshold change.

Analysis starts at `51aeb74` on `skylak33/confidence-aware-cluster-merging`, with
changes confined to the diagnostic script, its tests and this document. No
clustering, ECAPA, VAD, embedding, smoothing or production configuration is changed.

## Data contract and populations

The frozen moderate candidate is read from the previous experiment, without
alteration: centroid >=0.55, cross median >=0.20, q20 >=0.05, compactness drop
<=0.12, symmetric margin >=0.02, mutual neighbors, at least two members per
cluster, and a 32-member deterministic sampling cap. All feature values and gate
outcomes are reused from its saved trace.

Each cluster must contain only reliably assigned windows of one source speaker.
Two such clusters have a positive label if their source identities agree and a
negative label otherwise. Any mixed/unlabeled window or multiple identities
within either cluster makes the proposal ambiguous. No purity tolerance or
dominant-speaker substitution is used. Silence padding is allowed when speech
is attributable; overlapping embedding windows alone do not imply overlapping
reference speakers. Source turns are authoritative; attribution disagreement
with the rounded RTTM is marked uncertain. Rules were fixed before metrics.

The audit checks generated audio, source metadata, RTTM and embedding archive
hashes; source-utterance speaker/hash consistency; development split membership;
and source-turn/RTTM alignment. It verifies inherited source-content declarations,
not a new listening audit or rehash of every original source recording.

All 11,577 development windows have a single source speaker association. The 86
ambiguous pairs below arise from initial clusters containing windows of multiple
speakers, rather than mixed-window attribution. Synthetic speech-region timing
is VAD-assisted, not manually verified speech activity.

| Population | Candidates | Positive | Negative | Ambiguous | Recordings with pairs |
|---|---:|---:|---:|---:|---:|
| A: actual observations | 1,439 | 131 | 1,222 | 86 | 250 |
| A: first observation per membership pair | 1,439 | 131 | 1,222 | 86 | 250 |
| B: all initial nonempty-cluster pairs | 1,439 | 131 | 1,222 | 86 | 250 |

The corpus has 300 recordings, 323 source speakers and 887 source utterances.
Candidate clusters cover 315 source speakers. Positive opportunities occur in
85 recordings and involve 78 source speakers. Positive prevalence is 9.10% of
all pairs and 9.68% of the 1,353 unambiguous pairs.

**What the 1,439 proposals mean:** the implementation enumerates every unordered
cluster pair in each visited partition, before checking support, centroids or
nearest neighbors. These gates do not prefilter proposal generation. Here every
proposal is at step zero: the sole accepted merge goes from K=2 to K=1 and ends
the process. Consequently A=B, no initial pairs are omitted, and there are no
dynamic revisits. Candidate-generation failure is not the explanation in this
experiment. This does not establish the same result for future multi-step runs.

Membership labels were absent from the synthetic cache. Initial Fixed AHC labels
were reconstructed once on CPU from the existing embeddings. Every initial
pair's centroid, median, q20, dispersion, compactness loss and merged compactness
was checked against the saved trace with the existing `_pair_evidence`
calculator (absolute/relative tolerance 1e-12): all 1,439 pairs match. Saved feature
values, not recomputed values, enter the reported statistics. No frontend or
benchmark is rerun. The analysis replays saved accepted merges; references are
attached only offline and never forwarded to inference.

For general multi-step traces, observation IDs retain the global partition
state and pair memberships. Primary pair-level metrics keep the first encounter
of an unordered membership pair. Visit counts, later acceptance and raw event
metrics remain separate because alternatives and margins can change even when
the pair's members do not. Initial-pair opportunities are hypothetical pair
decisions, not independent false-negative merge actions: f fragments have
f(f-1)/2 positive pairs but need only f-1 connecting merges.

## Separability on unambiguous pairs

Directions are fixed in advance: larger centroid/median/q20/margin/support and
smaller dispersion/compactness loss are scored as safer. AUC below 0.5 is not
flipped. AP is step-weighted Average Precision, not trapezoidal PR area.

| Feature | ROC-AUC | AP | Positive median | Negative median | Histogram overlap |
|---|---:|---:|---:|---:|---:|
| Centroid similarity | 0.8926 | 0.5525 | 0.2854 | 0.0318 | 0.3832 |
| Cross median | 0.8862 | 0.4711 | 0.1734 | 0.0224 | 0.3714 |
| Cross q20 | 0.8695 | 0.4397 | 0.0933 | -0.0451 | 0.4077 |
| Cross IQR dispersion, lower safer | 0.4061 | 0.1117 | 0.1157 | 0.1071 | 0.7286 |
| Compactness loss, lower safer | 0.7510 | 0.2016 | 0.1138 | 0.2554 | 0.5522 |
| Symmetric margin | 0.8036 | 0.3153 | 0.0183 | -0.1223 | 0.4902 |
| Minimum cluster size | 0.1368 | 0.0711 | 3 | 8 | 0.3147 |

All features except margin use 131 positives/1,222 negatives. Margin is absent
for six positives and 39 negatives because K=2 provides no alternative: those
values are excluded, not imputed to zero. Undefined AUC/AP is preserved for empty
or one-class samples. Full q05/q25/q50/q75/q95 and IQR are in `feature_summary.csv`.
Histogram overlap is the sum of minima of class-normalized 20-bin histograms;
it is bin-dependent and is not an estimate of irreducible classification error.

Centroid/median/q20 Spearman correlations are 0.9890, 0.9663 and 0.9610. These
features largely repeat related geometric evidence. Dispersion does not behave
as a useful monotonic safety feature in the prescribed direction. Support's
inverse association reflects fragmented same-speaker clusters in this particular
initial AHC population, not a universal claim that small clusters are safe.
Individual-feature overlap does not prove that every possible joint rule fails;
no classifier or additional confidence estimator was fitted.

Three local figures are generated: empirical feature CDFs, combined ROC/PR curves,
and a Spearman correlation heatmap. No per-recording plots are produced.

## Frozen rule and rejection attribution

For A and B, selected acceptance and the conjunction of all gates coincide:
TP=1, FP=0, FN=130 pair opportunities, TN=1,222. There are 86 ambiguous proposals,
none accepted. Precision on unambiguous accepted pairs is 100% **from one event**;
recall is 0.7634%, acceptance 0.0695%, reference-label coverage 94.02%.
False merge rate FP/(TP+FP) and false-positive rate FP/(FP+TN) are both zero on
this development subset. Neither is a generalization guarantee.

| Gate | Rejected positives failing it | Negatives passing it |
|---|---:|---:|
| Centroid >=0.55 | 130 | 0 |
| Cross median >=0.20 | 90 | 30 |
| Margin >=0.02, or no alternative | 63 | 180 |
| Compactness drop <=0.12 | 58 | 218 |
| Mutual nearest neighbors | 54 | 216 |
| q20 >=0.05 | 40 | 161 |
| Support >=2 | 16 | 1,194 |

Counts overlap: 110 of 130 rejected positives fail multiple gates. Twenty
positives fail only the centroid gate. Eight negatives also fail only centroid.
Descriptively ignoring that gate while keeping the other gates would expose
21 positive, eight negative and one ambiguous initial pair; this is not a
proposed replacement rule or a dynamically executed merge sequence.

At the frozen point, centroid alone already passes exactly the one accepted
positive; the other conditions add no observed specificity beyond it. This does
not show that they are unnecessary after changing the operating point.

Each rejected positive retains its failed conditions and domain-normalized
numeric deficits. The closest numeric failed gate is median in 30 cases, margin
in 29, centroid in 26, compactness in 26 and q20 in 19. This convention compares
fractions of each feature's domain, not calibrated confidence or recommended
parameter changes. All pairs were evaluated; none was excluded before features.

## Development-only operating characteristics

Predefined targets are recall 10/25/50/75/90% and precision 90/95/99/100%. Reported
points are empirical ranking envelopes with tied scores kept together. Accepted
counts in the binary metrics refer only to unambiguous pairs; ambiguous and total
acceptance counts are also saved. Missing features stay excluded. No point is
selected as a production threshold.

For centroid ranking, a development score cutoff 0.4090716861 yields 13 known
positives, zero known negatives and **one ambiguous pair**: 14 total proposals
from 13 recordings, with 12 positive source identities. Conditional precision is
100%, recall 9.9237% and known-negative FPR zero. This does not establish a safe
region, especially after searching scores on the same development population.

At a predefined recall target of at least 25%, the best centroid envelope point
has recall 28.24%, precision 66.07%, 37 TP/19 FP and FPR 1.5548% (56 unambiguous
acceptances). At target recall >=50%, it has recall 52.67%, precision 58.97%,
69 TP/48 FP, FPR 3.9280% (117 unambiguous acceptances). Median, compactness loss,
margin and dispersion have no nonempty point reaching 95% empirical precision;
q20 and support reach it only through one accepted positive. Empty acceptance
has undefined precision, not 100% precision.

## Representative cases

- Rejected valid: `development_000223`, clusters 0/1, support 3, centroid 0.45443,
  median 0.22761, q20 0.18437, compactness loss 0.05093, margin 0.20070. Only
  centroid fails. `development_000243` is similar: centroid 0.44352, median
  0.23240, q20 0.18454, loss 0.06377, margin 0.06471, support 4.
- Negative near miss: `development_000095`, clusters 2/4, support 10, centroid
  0.38450, median 0.24667, q20 0.17698, loss 0.09465, margin 0.24645. All gates
  except centroid pass even though both pure clusters have different speakers.
- Ambiguous initial cluster: `development_000254`, clusters 0/2. Cluster A has
  eight windows from `VIVOSSPK45` and 11 from `VIVOSSPK35`; B has 17 from
  `id03512`. Centroid 0.44152, median 0.22781, q20 0.11808, loss 0.10224 and
  margin 0.14625 pass every gate except centroid. It is excluded from binary
  labels rather than assigned a convenient dominant identity.
- Feature overlap: positive `development_000240` 3/4 and negative
  `development_000180` 1/3 have centroid similarities 0.23017377/0.23017247.
  Their q20 values are 0.02361/0.07474, compactness losses 0.20344/0.15234 and
  margins -0.02031/0.04670: several safety features favor the negative example.
  These are overlapping/conflicting observations, not proof of intrinsic
  impossibility for all joint feature models.

The requested ViYT case is a separate frozen reused-test diagnostic.
`V-cuzXOX9sM` 0/2 passes all gates: centroid 0.55387, median 0.21075, q20 0.10591,
loss 0.01930, margin 0.03148, support 13/15. Its reliably assigned portions have
different speakers, but the clusters also contain nine/four uncertain windows.
Under the declared strict protocol its label is **ambiguous**, not a pure
negative. Previous diagnostics show immediate purity 0.90625 to 0.78125.
The rule measures embedding geometry, not reliable reference composition, so
this contaminated proposal can pass. It is not silently added to the
development FP count and is never used to choose an operating point.

## Support and uncertainty

Bins use the smaller cluster. The last row overlaps the size bins.

| Smaller cluster | Positive | Negative | Ambiguous | Positive prevalence among known pairs |
|---|---:|---:|---:|---:|
| Singleton | 16 | 28 | 1 | 36.36% |
| 2–4 windows | 91 | 184 | 9 | 33.09% |
| 5–15 windows | 22 | 983 | 74 | 2.19% |
| >=16 windows | 2 | 27 | 2 | 6.90% |
| Size ratio >=8 | 20 | 32 | 3 | 38.46% |

107/131 positive pairs involve a cluster of at most four windows. The current
minimum of two members excludes 16/131 positives (12.21%); 115/131 have enough
support and are almost all rejected elsewhere. Thus insufficient support is not
the principal cause of low recall. Lowering support alone would admit no new
positive under the remaining frozen gates. Small-cluster pairs are not
disproportionately negative in this conditional population, but their evidence
is sparse, overlapping windows are correlated, and class prevalence is not a
measurement of similarity-estimator reliability. Within the 2–4 and 5–15 bins,
centroid AUC remains 0.9203 and 0.9249. Large-cluster positives number only two.

Pairs are dependent within recordings and across reused source voices/utterances.
Connecting conversations that share a source speaker or content hash yields
19 components; the largest covers 282/300 recordings. The predeclared grouped
bootstrap requires >=10 components and no component exceeding half the corpus.
The dominance criterion fails, so no misleading pair-level or grouped CI is
reported. Pair-level AUCs do not represent 1,439 independent conversations.

Independent validation is unavailable. No validation/holdout labels or DataHive
labels enter this analysis, and no new data is downloaded. ViYT is reused-test
evidence. Source namespaces do not establish speaker-disjointness from ECAPA
training or identity equivalence across datasets. The clean synthetic corpus
does not exercise the natural overlap/mixed-window conditions in the unsafe case.

## Recommended next step

Choose **D: obtain independent natural development data**, with auditable speaker
identities and overlap annotations. The current features have useful ranking
signal, so this experiment does not justify abandoning merging for an unrelated
clustering algorithm. However, high precision at useful coverage and robustness
to contaminated clusters remain unproven. Independent data is the priority
before changing the criterion or claiming a safe region. No next algorithm is
implemented here.

## Reproduction and verification

```powershell
python experiments/analyze_merge_separability.py `
  --source results/confidence_aware_merging `
  --output results/merge_separability
python -m pytest -q
python -m compileall -q diarize eval experiments tests
git diff --check
```

Fresh extraction refuses an existing output directory. `--refresh-derived`
reuses candidate CSVs to regenerate the small rejection/operating-point/example
tables and verify stored features; it does not recluster or perform inference.
Its source hash and input hashes are saved separately in `derived_provenance.json`.

Outputs include raw/initial/unique candidate populations, restored memberships,
provenance, full feature quantiles, confusion matrices, rejection conditions and
deficits, support slices, operating points including ambiguous acceptance,
representative cases, the separate ViYT case, summary JSON and three figures.
Generated artifacts and embedding caches are excluded from Git. Frozen clustering
SHA256 remains `241132be61056e398dec0e85b774c3ade9fa47260350cfe945a6f286f445cf13`;
checkpoint SHA256 remains
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.

Final checks: 370 tests passed (348 existing plus 22 new), zero failures/skips;
compileall and diff-check succeeded. A path-scoped comparison against `51aeb74`
confirmed no changes to `diarize/`, `models/`, `configs/`, `eval/` or `utils/`.
