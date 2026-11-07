# ViYT-Diar Evaluation Protocol Audit

Audit date: 2026-11-02. This document distinguishes published facts from local
observations. It does not redefine the benchmark or repair its annotations.

## Confirmed facts

- The official dataset card describes ViYT-Diar as a test-only benchmark with
  100 complete audio rows. The audio is mono, 16 kHz WAV sourced from public
  Vietnamese YouTube material. The annotations were created manually in Label
  Studio, exported as RTTM, and then stored as speaker/start/end lists in the
  published Parquet file.
- The MAPR 2025 paper describes 100 mono recordings of approximately 60 seconds.
  Its annotation criteria require clear speaker voices and merge silence shorter
  than 250 ms between segments.
- The paper states that DER uses a 0.25 second collar and includes overlapping
  speech. Its tables also report a separate "ignore overlap" condition. For the
  paper's primary ViYT-Diar result, the overlap-aware value is 2.38%.
- The official Hugging Face repository currently contains the dataset card,
  `.gitattributes`, and one Parquet shard. It does not contain a UEM file or
  another published evaluation-region manifest.
- The public documentation does not identify known partially annotated
  recordings and does not name the scoring implementation or its exact version.

Primary sources:

- [Official ViYT-Diar dataset card](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar)
- [Official dataset repository files](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar/tree/main)
- [MAPR 2025 paper and project page](https://v-nhandt21.github.io/publication/2025-wavlm-diarization)

## Targeted data audit

The published Parquet annotations and decoded audio lengths were inspected for
the three recordings previously associated with very large false-alarm values:

| Recording | Audio duration (s) | First reference start (s) | Last reference end (s) | Published labels |
|---|---:|---:|---:|---:|
| `C4HFERLpRfI` | 62.710 | 57.415 | 62.378 | one (`Speaker_2`) |
| `Uaq5N3FbQb4` | 60.054 | 43.122 | 58.980 | one (`Speaker_2`) |
| `ed0Vt0NxJUI` | 62.749 | 28.381 | 57.300 | one (`Speaker_2`) |

The timestamps are within the decoded audio bounds, so there is no simple global
time-base mismatch in these rows. They nevertheless leave long leading regions
without reference speech. A local Silero VAD pass detected substantial activity
before each first reference segment, including activity near time zero. VAD is
not ground truth, especially for music and noise, but this independently flags
the leading regions for human review.

Manual listening notes supplied during this audit report speech at 0:00 in all
three files and speaker changes near 0:12, 0:36, and 0:09 respectively. Those
notes are evidence of a likely annotation-coverage problem, but they have not
been confirmed by a second human annotator and must not be converted into
reference labels automatically.

## Current scorer comparison

With no external UEM, `eval.der` scores the full decoded audio duration when a
duration is supplied. The default collar is 0.25 seconds and overlap is included
unless `--skip_overlap` is set. These settings match the protocol statements in
the paper. The exact published score cannot yet be claimed reproducible because
the paper does not disclose an official UEM, scorer/version, label conversion,
or any recording-specific exclusions.

External UEM support is deliberately opt-in. A supplied UEM is validated and
used exactly; invalid or incomplete UEM input fails instead of silently falling
back to full-audio scoring. In the absence of an author-provided UEM, the
full-audio result remains the benchmark-facing result and any reviewer-created
UEM result must be labeled as an audit experiment.

## Unresolved questions

1. Is every clear speech event intended to be annotated, including laughter,
   speech over music, and speech in the leading part of an extracted clip?
2. Were the three rows intentionally annotated only over a selected region?
3. Was an unpublished UEM or trimming step used for the paper's results?
4. Which DER implementation/version and collar convention produced the tables?
5. Were overlapping annotations preserved through RTTM-to-Parquet conversion?

## Recommended validation

1. Ask the dataset authors for the original RTTM, scoring command, dependency
   versions, and any UEM or trimming manifest used for the paper.
2. Have two reviewers independently mark speech/non-speech and speaker changes
   for the three recordings, then adjudicate disagreements. Review laughter and
   speech over background music explicitly.
3. Keep proposed corrections and reviewer UEM files as versioned, hashed audit
   artifacts. Do not overwrite the published reference.
4. Report published-reference/full-audio and reviewer-audited measurements
   separately until the dataset authors confirm the intended protocol.
5. Use a separate development set for clustering decisions and a previously
   unused held-out set for final evaluation.

## Project manual-audit reference

The reviewer subsequently validated speaker turns for `C4HFERLpRfI`,
`Uaq5N3FbQb4`, and `ed0Vt0NxJUI`. The preserved RTTM files are under
`audit/viyt_manual_reference_20261102`. They are accepted for local analysis but
remain separate from the published ViYT-Diar reference; derived metrics must be
labeled accordingly.
