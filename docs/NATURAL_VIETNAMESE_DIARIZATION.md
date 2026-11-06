# Natural Vietnamese diarization development benchmark

## Stage A status

This document records the repository and remote-metadata audit performed on
2026-11-06. No audio was downloaded, no reference was generated, and no
diarization result was measured in this stage. Dataset revisions are pinned in
`configs/natural_vietnamese_benchmark.json`; counts below must be revalidated
against those revisions during acquisition.

The benchmark decision is whether either frozen Fixed AHC threshold (the
production `0.65` or synthetic-benchmark `0.75`) is competitive with the four
existing automatic-count alternatives on natural Vietnamese conversations.
Initial results will not tune any threshold or change a production default.

## Stage B acquisition

Stage B added `experiments/acquire_natural_vietnamese_benchmark.py`. The CLI
pins both repository revisions from the config, filters records before
acquisition, uses the Hugging Face cache for resumable/reused files, compares
downloaded size and SHA-256 with Hub LFS metadata, and writes a deterministic
manifest under the ignored `results/` tree. It supports `--download`,
`--metadata-only`, `--cache-dir`, `--offline`, `--dataset`, and
`--max-recordings`.

The metadata-only smoke inspected all configured metadata without downloading
audio and found the expected 8 DataHive and 23 Humyn Vietnamese records. The
download smoke used:

```powershell
python experiments\acquire_natural_vietnamese_benchmark.py `
  --download --dataset datahive --max-recordings 1 `
  --cache-dir D:\Downloads\natural-vietnamese-diarization\hf-cache `
  --output results\natural_vietnamese_benchmark\datahive_smoke_manifest.json
```

`vi_000001.mp3` was the only downloaded audio file: 12,218,540 bytes, official
LFS SHA-256
`73db5cf99405944368beffdb8994033438e67c43dea5a649f2f5205ba0328c94`,
48 kHz stereo MP3, and 763.584979 decoded seconds. The publisher metadata says
763.632 seconds, a 0.047021 second difference that reference validation must
allow for MP3 timing. Repeating the same command with `--offline` completed
from cache without a network request. The Humyn 1.027 GiB shard was not
downloaded; its 23 Vietnamese rows were inspected through non-audio Parquet
columns only.

## Stage C reference annotation

Stage C added `experiments/prepare_natural_vietnamese_references.py`. DataHive
audio is decoded without collapsing its two channels, Silero VAD is run on
each channel independently, and the shared timeline is written to RTTM with
stable `SPEAKER_00` (left/caller) and `SPEAKER_01` (right/callee) identifiers.
Intervals from the two speakers are not partitioned or otherwise changed, so
reference overlap remains present. The resulting provenance is explicitly
`silver_reference`, never human ground truth.

The complete pinned DataHive set was cached under
`D:\Downloads\natural-vietnamese-diarization\hf-cache` and prepared with:

```powershell
python experiments\prepare_natural_vietnamese_references.py `
  --dataset datahive `
  --acquisition-manifest results\natural_vietnamese_benchmark\datahive_acquisition_manifest.json `
  --output-dir results\natural_vietnamese_benchmark\datahive_references
```

All 8 recordings were eligible. Their 4,147 RTTM rows contain both expected
speakers and 418.0 seconds of preserved two-speaker overlap. Channel speech
totals sum to 7,094.685 seconds. The configured QC checks found no high
cross-channel correlation, high-overlap ratio, severe participation imbalance,
or leakage-RMS threshold violation. This is an automatic screen, not a manual
quality judgment; the QC thresholds are heuristic review triggers, not tuned
production cutoffs. The review manifest is deterministically seeded and has 435
simultaneous-speech intervals of at least 0.5 seconds plus 40 ordinary
non-overlap intervals. Its `human_verified` field remains `false`.

The Humyn parser accepts only non-empty timestamped JSON lists, preserves
cross-speaker overlap, assigns speaker IDs by first occurrence, and rejects
malformed timestamps, invalid/missing speakers, duplicates, non-positive
segments, and intervals outside decoded audio duration. The 20 Vietnamese
plain-text records are excluded as `unsupported_annotation_format`. The three
timestamped two-speaker records (`viet_1.wav`, `viet_2.wav`, and `viet_3.wav`)
remain excluded as `audio_shard_not_cached`: boundary validation requires the
separately gated 1.027 GiB shard. No Humyn reference is claimed as eligible in
Stage C.

## Stage D benchmark integration

Stage D added `experiments/run_natural_vietnamese_benchmark.py` as a thin
natural-data adapter around the existing embedding extraction, clustering,
production postprocessing, RTTM, partition metrics, and DER implementation.
It reads the frozen configuration list directly from
`configs/natural_vietnamese_benchmark.json`; there is no threshold-search CLI
and no reference speaker count is passed to an automatic clustering method.

For each recording the runner verifies the audio, reference, annotation
manifest, configuration, checkpoint, and relevant code hashes. It then
extracts embeddings once for each of the two declared modes and reuses them for
all six clustering configurations. `clustering_only` uses the union of
reference speech regions; `full_pipeline` uses the shared mono-mix Silero VAD
settings. Each hypothesis is scored twice without regeneration: overlap
`included` and overlap `excluded`, both with the configured 0.25-second collar.
DER is pooled by reference speaker-seconds, while per-recording DER, component
seconds, speaker counts, ARI/NMI, purity, fragmentation, runtime, embedding
hashes, hypothesis hashes, and clustering traces remain available separately.

A real one-recording CUDA smoke on `vi_000001` loaded the pinned checkpoint
with SHA-256
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.
It produced 757 full-pipeline and 751 clustering-only embedding windows, 12
distinct hypothesis RTTMs (six configurations by two modes), and 24 protocol
rows after the two overlap conditions. Embedding hashes were identical across
all six configurations within each mode, and included/excluded scoring reused
the same hypothesis hash. The smoke metrics are integration evidence only and
are not reported as a benchmark result. The eight-recording Stage E evaluation
has not run.

## Repository audit

The natural benchmark should extend the existing synthetic benchmark rather
than add another diarization implementation:

| Concern | Reusable code | Required Stage B--D change |
|---|---|---|
| Audio/VAD/ECAPA | `diarize/vad.py`, `diarize/pipeline.py` | Add dataset preparation around the existing frontend. |
| Clustering | `diarize/cluster.py` | None for the frozen comparison. |
| Shared embeddings | `diarize.pipeline.diarize` and `run_synthetic_diarization_benchmark.py` | Cache once per recording and evaluation mode. |
| RTTM and DER | `eval/der.py` | Reuse overlap-aware pyannote scoring and pooled-second aggregation. |
| Provenance | `utils/provenance.py` | Record revision, input/reference/checkpoint/code hashes. |
| Benchmark rows | `experiments/run_synthetic_diarization_benchmark.py` | Generalize the manifest input and add Fixed AHC `0.65`. |
| Analysis | `experiments/analyze_synthetic_benchmark.py` | Add recording bootstrap, both overlap conditions, and dataset strata. |
| Corpus validation | `experiments/prepare_diarization_devset.py` | Reuse its path, hash, RTTM, and leakage checks after preparation. |

`requirements.txt` already contains `datasets`, `huggingface_hub`, `pyarrow`,
`soundfile`, and `requests`; the acquisition stage needs no new dependency.
Generated files under `results/` are already excluded by `.gitignore`.

## Dataset audit

The audit used the Hub API, Dataset Viewer metadata, and selected non-audio
Parquet columns. It did not rely only on dataset-card prose.

### DataHive Multilingual Conversations

- Repository: <https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample>
- Pinned revision: `c29728a0c9792ba486807fa1ea9ca64e292cec01`
- License: CC BY 4.0
- Repository total: 85 conversations, 1.08 GB as reported by the Hub.
- Vietnamese selection: 8 `language == "vi"` records, 12,427.824 seconds
  (3.452173 hours), 198,848,608 bytes (189.637 MiB).
- Vietnamese files: `preview/vi_000001.mp3` through
  `preview/vi_000008.mp3`; all metadata rows declare `languages == ["vi"]`
  and both countries as `VN`.
- Audio contract from the publisher: stereo MP3, 48 kHz, caller on the left,
  callee on the right, shared timeline.
- Automatic metadata reports 6.0--15.0% overlap for these eight recordings.
  These values are planning diagnostics, not reference annotations.

The eight conversations are all nominally two-speaker. The metadata exposes no
speaker identity, so repeated people across recordings cannot be ruled out and
speaker-disjointness cannot be claimed. Channel assignment enables a useful
silver reference, but channel leakage, echo, background voices, and VAD errors
must be audited before a recording becomes eligible.

### Humyn Asian High-Fidelity ASR

- Repository: <https://huggingface.co/datasets/humyn-labs/Asian-High-Fidelity-ASR-Dataset>
- Pinned revision: `9d640c7c6a94cd21641d0934591da7870e00c9b5`
- License: CC BY 4.0
- Repository total: 242 rows and 1.9 GB as reported by the Hub.
- Schema: `language: string`, `file_name: string`, `audio: Audio(16000)`,
  `transcript_json: string`, and `type: string`.
- Vietnamese selection: 23 rows: 3 `multi-speaker` WAV recordings and 20
  `single-speaker` MP3 recordings.
- The three multi-speaker rows contain valid JSON lists with 431 total
  timestamped segments, exactly `Speaker 1` and `Speaker 2`, finite increasing
  timestamps, no missing labels, and no exact duplicate segments. Their last
  annotated endpoints total 2,069.422 seconds (34.490 minutes). This is an
  annotation-span lower bound, not a measured total audio duration.
- One conversation contains 5.575 seconds of overlap between consecutive
  transcript segments; the other two contain none by that limited check.
- All 20 single-speaker rows contain plain transcript text rather than JSON or
  timestamped segments. They cannot directly yield speech-boundary RTTM despite
  the column name `transcript_json`.

Only shards `train-00000-of-00011.parquet` and
`train-00010-of-00011.parquet` contain Vietnamese rows. They total
1,119,709,076 bytes (1.043 GiB), versus 1.768 GiB for all 11 shards.
The first shard alone is 1,102,412,810 bytes (1.027 GiB) and contains all three
Vietnamese conversations plus 19 non-Vietnamese rows because audio is embedded
in Parquet. The Dataset Viewer filter/rows endpoint currently cannot scan this
large row group, so a robust official-tooling path cannot promise row-level
audio download without first fetching that shard. Stage B must verify this with
a smoke run before claiming a smaller transfer.

The structured multi-speaker transcripts are reviewed ASR annotations, not
documented diarization boundary adjudication. Their RTTM provenance will be
`transcript_derived_reference`, never human-verified ground truth. The 20
single-speaker recordings can only be used as a separate K=1 overfragmentation
slice if automatic VAD produces `silver_reference` speech regions.

## Storage and runtime gate

| Acquisition | Required transfer | Approval status |
|---|---:|---|
| DataHive metadata only | 22,493 bytes | Inspected in Stage A |
| DataHive Vietnamese audio | 189.637 MiB | Awaiting Stage B approval |
| Humyn metadata columns | Range reads only | Inspected in Stage A |
| Humyn required shards | 1.043 GiB | Awaiting explicit approval |
| Humyn full repository | 1.768 GiB | Not required |

Channel extraction will be streamed; keeping two decoded 16 kHz PCM channels
would add about 759 MiB for DataHive, so intermediate WAV files will not be
persisted unless debugging requires a bounded sample. References, manifests,
plots, and embeddings remain under the ignored `results/` tree.

The prior local GPU synthetic runs spent about 1.7 minutes of shared frontend
time per input audio-hour for both evaluation modes combined. Extrapolation to
DataHive is roughly six minutes for ECAPA/VAD, but this is not a guarantee:
natural MP3 decoding and long speech regions differ. Clustering is the larger
uncertainty. At a 0.75 second step, the 51-minute DataHive recording can produce
about 4,100 windows and roughly 16.8 million pairwise entries; dense spectral
methods may dominate time and memory. Stage D must benchmark one short and the
longest eligible recording before publishing a full-run ETA. CPU runtime is
not estimated without that measurement.

## Reference and quality-control plan

DataHive preparation will decode without changing the shared timeline, run the
existing Silero VAD independently on left and right channels, assign stable
recording-local IDs (`SPEAKER_00` left and `SPEAKER_01` right), and retain true
simultaneous intervals in RTTM. Every reference will be labeled
`silver_reference` with dataset revision, source hash, decode parameters, VAD
version/configuration, and reference hash.

Eligibility is fail-closed for unreadable audio, a channel count other than two,
invalid duration/sample rate, missing channel speech, invalid intervals, or an
RTTM interval outside the audio duration. Other signals are review flags, not
silent exclusions: per-channel RMS/peak, clipping and silence, VAD duration and
balance, overlap ratio, correlated simultaneous activity, and likely leakage or
echo. The review manifest will contain every flagged interval plus a seeded
ordinary sample. No item becomes `human_verified` without an explicit review
record.

Humyn preparation will parse only JSON list/dict formats explicitly supported
by tests. It will reject rather than repair invalid JSON, absent speaker IDs,
non-finite or reversed timestamps, out-of-bounds intervals, duplicates, and
inconsistent speaker mapping. Multi-speaker and VAD-derived single-speaker
references remain separate provenance strata.

## Frozen evaluation protocol

The six configurations in the JSON config are Fixed AHC `0.65`, Fixed AHC
`0.75`, legacy adaptive AHC, robust adaptive AHC, spectral, and NME-SC. They
share decoded audio, VAD, segmentation, checkpoint, embeddings, seed, and
postprocessing within each recording/mode. No clustering code or production
default changes before the initial report.

The main `full_pipeline` condition uses normal mono-mix VAD. A diagnostic
`clustering_only` condition uses the union of reference speech regions and is
reported separately because it benefits from reference timing. Both use the
current 1.5/0.75 second ECAPA windows and checkpoint SHA-256
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.

Scoring uses `pyannote.metrics` through `eval/der.py`, a 0.25 second collar,
the complete recording as UEM, and pooled (micro) DER from scored seconds.
Every method is scored twice: overlap included as the primary result and
reference overlap excluded as a diagnostic. The current hypothesis pipeline
emits one speaker at a time, so overlap-included DER includes its inability to
represent two simultaneous speakers; this limitation will not be removed from
the primary reference. Report DER, Miss, False Alarm, SpeakerError, count
accuracy/MAE, ARI on single-speaker-compatible windows, runtime, and every
per-recording row.

Uncertainty uses paired recording-level bootstrap resampling (10,000 draws,
seed 42) and 95% percentile intervals. Dataset/provenance strata are reported
separately before any combined number. With only eight DataHive and three
timestamped Humyn conversations, intervals will be wide and small deltas will
not justify a production change. Almost all usable natural conversations have
K=2; the experiment cannot establish K=3--5 behavior. Humyn K=1 can only test
fragmentation under a different reference process.

## Implementation plan

1. **Stage B -- acquisition (complete).** The repository-convention CLI has
   `--download`, `--cache-dir`, `--offline`, `--dataset`, `--metadata-only`, and
   `--max-recordings`; it pins revisions, uses Hub cache/resume behavior,
   selects the eight DataHive paths and two Humyn shards, verifies Hub/LFS
   metadata plus local SHA-256, and passed metadata-only, download, and offline
   smoke tests. The Humyn 1.027 GiB shard still requires approval.
2. **Stage C -- references (complete for cached data).** Channel-VAD RTTM,
   strict Humyn timestamp parsing, provenance, exclusion/QC manifests, and
   deterministic review samples are implemented and covered by synthetic CI
   fixtures. All eight cached DataHive recordings are eligible; Humyn remains
   fail-closed without its audio shard.
3. **Stage D -- benchmark (complete).** The natural runner consumes the
   prepared annotation manifest, includes Fixed AHC `0.65`, reuses each
   mode-specific frontend across all frozen methods, and scores both overlap
   conditions without changing clustering or the hypothesis.
4. **Stage E -- evaluation.** Run only eligible frozen-revision recordings,
   preserve all exclusions, measure the short/long runtime smoke before the
   complete GPU run, and keep generated artifacts outside Git.
5. **Stage F -- analysis.** Produce dataset-stratified aggregates, paired
   Fixed-versus-Robust deltas and confidence intervals, count confusion,
   distributions, failure cases, and a natural-versus-synthetic limitations
   table. Any optional threshold selection is a separately labeled experiment;
   this corpus is currently too small to treat it as production validation.

The remaining seven DataHive files were approved and cached. A separate
decision is still required before downloading the 1.027 GiB Humyn shard for
its three potentially eligible conversations. Stage E has not started, and no
result in this document is a full-corpus DER measurement.
