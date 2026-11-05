# Synthetic Vietnamese diarization benchmark

## Scope

This benchmark tests whether automatic speaker-count and clustering behavior
observed on the reused ViYT-Diar test set generalizes to different Vietnamese
voices. It uses real recorded speech to construct deterministic conversations;
it does not claim that 600 generated WAV files are 600 independent natural
conversations.

The primary benchmark is deliberately clean: no added noise, reverberation,
channel transform, or overlapping speech. Short-turn, acoustic-change, and
overlap conditions remain separate future stress tests.

## Local source audit

No dataset was downloaded by the benchmark code.

| Source | Local utterances | Speakers | Duration | Status |
|---|---:|---:|---:|---|
| VIVOS (`vietnamese_asr`) | 12,420 | 65 | 15 reported hours | complete 4/4 Parquet shards |
| VieSpeaker Opus | 319,296 | 4,101 | not measured locally | partial 96/110 shards |
| DNU-ViSpeech | 31,070 | 26 | 75.569759 hours | complete, WAV headers validated |

Speaker IDs are namespaced by dataset. DNU official speaker-disjoint splits are
preserved. VIVOS test speakers are holdout-only; VIVOS train speakers are split
between development and validation. VieSpeaker speakers are assigned by a
stable SHA-256 rule. Unknown cross-dataset real-person overlap remains a
limitation because namespaced IDs do not prove distinct physical identities.

The frozen ECAPA checkpoint is
`58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.
Repository evidence points to Dolly/VLSP training code, but a complete epoch-34
training manifest is unavailable, so overlap with these three sources cannot be
ruled out conclusively.

## Generation protocol

`configs/synthetic_vietnamese_benchmark.json` fixes seed 42, 16 kHz mono audio,
30--45 second duration, 3--8 second turns, 0.3--1.0 second inserted silence, and
speaker counts 1--5. Each selected speaker occurs at least twice. Source
datasets rotate deterministically so that no large corpus dominates selection.

Silero VAD is loaded once and selects continuous source-speech regions. RTTM is
therefore VAD-assisted reference activity, not manual ground truth. The exact
source utterance IDs and hashes are retained in each conversation JSON.

The resulting clean corpus contains:

| Split | Conversations | Unique speaker IDs | Source utterances | Duration |
|---|---:|---:|---:|---:|
| Development | 300 | 323 | 887 | 3.0092 h |
| Validation | 150 | 162 | 430 | 1.5057 h |
| Holdout | 150 | 159 | 422 | 1.4934 h |

Each split is balanced by reference speaker count: 60 conversations per count
on development and 30 per count on validation and holdout. Validation confirmed
that speaker IDs, source utterances, and decoded source-content hashes do not
cross splits. Generated audio, RTTM, embeddings, and experiment outputs are
ignored by Git.

## Evaluation protocol

Two modes are reported separately:

- `clustering_only`: reference speech regions feed the production sliding-window
  embedding code. True speaker identities are never passed to clustering.
- `full_pipeline`: Silero VAD supplies speech regions before the same ECAPA,
  windowing, clustering, and temporal postprocessing.

Within a mode and recording, ECAPA embeddings are extracted once and shared by
all five methods. Scoring uses pooled DER with a 0.25 second collar, overlap
included, and no UEM. Partition metrics use only embedding windows touched by
exactly one reference speaker.

The five automatic-count configurations are:

1. Fixed AHC, threshold 0.75.
2. Legacy adaptive AHC, fallback threshold 0.75.
3. Robust adaptive AHC, fallback threshold 0.75.
4. Spectral clustering.
5. NME-SC.

The production default threshold remains unchanged.

## Measured aggregate results

### Clustering-only DER / count accuracy

| Method | Development | Validation | Holdout |
|---|---:|---:|---:|
| Fixed AHC | 5.3931% / 70.33% | 5.3476% / 67.33% | **4.0569% / 75.33%** |
| Legacy adaptive | 8.2314% / 59.33% | 7.1693% / 62.67% | 6.5984% / 66.67% |
| Robust adaptive | 9.9193% / 55.67% | 8.2582% / 60.00% | 7.9174% / 60.00% |
| Spectral | 8.4692% / 62.67% | 7.7760% / 58.67% | 7.6604% / 65.33% |
| NME-SC | 19.1439% / 60.33% | 24.4391% / 52.00% | 19.4409% / 60.67% |

### Full-pipeline DER / count accuracy

| Method | Development | Validation | Holdout |
|---|---:|---:|---:|
| Fixed AHC | 6.0574% / 69.33% | 6.5890% / 64.00% | **4.6302% / 74.67%** |
| Legacy adaptive | 9.1993% / 57.67% | 7.3907% / 62.67% | 6.8869% / 66.67% |
| Robust adaptive | 10.6547% / 55.00% | 8.4129% / 60.00% | 8.6090% / 61.33% |
| Spectral | 8.8421% / 63.67% | 8.1273% / 62.67% | 8.5106% / 65.33% |
| NME-SC | 17.9660% / 63.00% | 22.5918% / 52.00% | 20.8934% / 58.00% |

Fixed AHC was selected on development and frozen before validation and the
single holdout run. Robust adaptive AHC does improve the holdout K=2 subgroup
(Mode A DER 0.9298% versus 3.6479%), but it fragments one-speaker recordings
(18.5539% versus 3.2005%) and degrades K=4--5. On holdout paired comparisons
across both modes, Robust improves 9 cases, worsens 60, and leaves 231 unchanged.

Post-hoc development-only sensitivity runs used 25 balanced conversations for
each additional seed. Fixed versus Robust DER was 4.6384% versus 9.8844%
(clustering-only) and 5.0104% versus 6.0914% (full pipeline) for seed 123;
for seed 2026 it was 4.5793% versus 6.7569% and 5.9528% versus 7.9372%.
These small runs confirm the direction, but are not substitutes for full-size
grouped resampling and were not used to alter the frozen selection.

The worst holdout case is `holdout_000021`: Fixed returns one speaker with 0%
DER, while Robust selects an apparent hierarchy-gap outlier and returns four
clusters in clustering-only mode (58.70% DER) and seven final speakers in the
full pipeline (72.02% DER). The fixed-threshold fallback would have returned one
speaker. This is count-estimation fragmentation, not evidence of a VAD failure.

## Reproduction

Use the existing virtual environment; do not call a remote dataset loader.

```powershell
$documents = [Environment]::GetFolderPath('MyDocuments')
$python = Join-Path $documents 'diarize-vietnamese-speech\.venv\Scripts\python.exe'

& $python experiments\audit_local_speaker_datasets.py `
  --vietnamese-asr "$documents\vietnamese_asr" `
  --viespeaker "$documents\VieSpeaker-Dataset-opus" `
  --dnu-vispeech "$documents\DNU-ViSpeech"
& $python experiments\generate_synthetic_conversations.py

& $python experiments\run_synthetic_diarization_benchmark.py `
  --conversation-json results\synthetic_vietnamese_benchmark\generated `
  --model exps\exp1\model\model_0034.model `
  --split development --device cuda `
  --output-dir results\synthetic_vietnamese_benchmark\experiments\development

& $python experiments\analyze_synthetic_benchmark.py `
  results\synthetic_vietnamese_benchmark\experiments\development\recording_results.json
```

Run validation only after recording the development selection, and run holdout
once after the configuration is frozen.

## Interpretation limits

- Conversation construction and speech activity are automatic.
- Many conversations reuse speakers and source utterances within one split, so
  per-conversation rows are statistically dependent.
- VieSpeaker is a partial local mirror and uses lossy Opus audio.
- Speaker labels were not independently verified by listening to every file.
- DNU has only four speakers in validation and four in holdout.
- The clean benchmark does not cover overlap, short turns below three seconds,
  noise, reverberation, or natural conversational interaction.
- Results support Fixed AHC 0.75 for this benchmark, not an automatic change to
  production defaults and not a universal production claim without a natural
  independent development corpus.
