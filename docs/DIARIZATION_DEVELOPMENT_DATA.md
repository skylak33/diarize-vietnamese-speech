# Independent diarization development data

## Current status

Empirical clustering and smoothing selection is **blocked**. The local inventory
contains the 100-recording ViYT-Diar corpus and derived references/hypotheses,
but that test set has already been used for exploratory analysis. It remains
valid only for historical reproduction and labeled post-hoc diagnostics. The
local VLSP material is verification/train speech with trial or speaker labels,
not conversational recordings with time-aligned speaker turns. Neither source
is an independent diarization development set.

The preparer deliberately has no discovery or fallback mode. Running it without
an explicit inventory fails rather than treating ViYT-Diar, VLSP, generated
RTTM, or approximate listening notes as development truth.

## Recommended acquisition

Use the official **VoxConverse v0.3 development set** for the first independent
clustering/postprocessing study. Its maintainers provide WAV recordings and
RTTM annotations, label the repository as v0.3 after correcting test RTTMs, and
release the dataset for research under CC BY 4.0 while noting that copyright in
the source videos remains with their owners:

- [Official VoxConverse repository](https://github.com/joonson/voxconverse)
- [Official VoxConverse dataset page](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)

Do not acquire it until storage, access, and license acceptance are confirmed.
After approval:

1. Download only the official development WAV archive. Clone or download the
   v0.3 annotation repository and record the exact Git commit. Leave the
   official test audio and RTTM untouched for a later, one-time evaluation.
2. Preserve the original recording ID. Record the parent YouTube/source ID and
   conversation/event ID separately. If several clips come from one source or
   conversation, keep the whole group in one split.
3. Divide the official development recordings into `development` (calibration)
   and `validation` partitions by source/conversation groups, approximately
   70/30. Balance duration and reference speaker-count ranges only after the
   groups are fixed; do not move clips independently to improve balance.
4. VoxConverse RTTM labels should be treated as recording-local unless the
   maintainers provide an explicit cross-recording identity mapping. Set
   `global_speaker_ids` to `false`; the manifest will report speaker-disjointness
   as not checkable. If a reliable mapping is obtained, set it to `true` and let
   the validator reject any identity crossing the split.
5. Create the input inventory below, run the preparer, review all counts and
   hashes, then freeze both the inventory and output manifest before tuning.

AMI is a secondary option when meeting acoustics are preferred. The official
AMI corpus contains about 100 hours of English meetings, CC BY 4.0 signals and
manual annotations, and published train/development/evaluation meeting sets.
However, its official manual annotations are NXT rather than RTTM, so a
versioned, reviewed NXT-to-RTTM conversion would be required before this tool
can accept it:

- [Official AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [Official AMI download and annotation revisions](https://groups.inf.ed.ac.uk/ami/download/)
- [Official AMI partitions](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml)

## Domain mismatch

VoxConverse contains in-the-wild YouTube speech, including political debates and
news segments. AMI contains English meetings recorded in instrumented rooms.
Neither represents Vietnamese conversational speech, its phonetics, typical
channels, or the ViYT-Diar source distribution. These corpora can support
development of language-agnostic hierarchy stopping and temporal consistency,
but an improvement on either corpus is not evidence that Vietnamese DER will
improve. Before changing defaults, confirm the frozen candidate on a newly
acquired Vietnamese conversational set with authoritative turn annotations.

## Input contract

The JSON inventory contains dataset provenance and one row per recording:

```json
{
  "dataset": {
    "name": "VoxConverse",
    "source": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/",
    "license": "CC-BY-4.0 (source-video copyright retained by owners)",
    "revision": "v0.3; git commit <full SHA>",
    "language": "record as verified from acquired metadata",
    "domain": "in-the-wild YouTube debates/news",
    "annotation_format": "RTTM",
    "project_usage": "independent_development",
    "global_speaker_ids": false
  },
  "root": "C:/path/to/frozen/voxconverse-dev",
  "recordings": [
    {
      "recording_id": "example",
      "split": "development",
      "audio_path": "audio/example.wav",
      "rttm_path": "rttm/example.rttm",
      "source_id": "original-source-id",
      "conversation_id": "conversation-or-event-id"
    }
  ]
}
```

Only `development` and `validation` are accepted, and both must be present.
Paths must be relative to `root`. Audio must be readable WAV; RTTM rows must use
the matching recording ID, valid finite times, and fit inside the audio. The
validator requires at least one multi-speaker recording, rejects duplicate
recording/audio/reference content, and rejects source or conversation overlap
between splits. When global speaker IDs are declared, it also rejects speaker
overlap.

Run:

```text
python -m experiments.prepare_diarization_devset inventory.json --output frozen_manifest.json
```

The output records SHA-256 for every WAV and RTTM, audio duration, turn and
speaker counts, split/corpus totals, source/license/revision metadata, and the
result of each leakage check. The tool validates and fingerprints files; it
does not download, copy, relabel, or synthesize annotations.

## Evaluation boundary

Tune AHC count estimation and smoothing only on `development`. Compare the
frozen choice once on `validation`, reporting pooled DER components, macro DER,
speaker-count accuracy/MAE, overcount/undercount rates, runtime, and automatic
versus oracle-count diagnostics. Oracle counts remain diagnostic only. Preserve
the official VoxConverse test set and any future Vietnamese test set until all
algorithm and parameter choices are frozen.
