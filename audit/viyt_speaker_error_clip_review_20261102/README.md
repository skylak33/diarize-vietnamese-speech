# ViYT-Diar SpeakerError clip review — 2026-11-02

This audit preserves the reviewer-validated content of four clips selected from
the largest `SpeakerError` intervals. Clip-relative labels `speaker 1` and
`speaker 2` identify voices only within each reviewed clip; they must not be
matched to published RTTM labels by their numbers.

The review clips contain one second of context around the scorer interval. The
absolute timestamps below are obtained by adding the clip start shown in the
filename. Background music is noted separately and is not a speaker.

## Manual annotations

| Recording and clip | Clip-relative annotation | Source-audio annotation |
|---|---|---|
| `98EV6H0ZRlU_9.410-14.320.wav` | speaker 1 laughter at 0.000; speaker 2 at 0.000–3.000; speaker 1 at 3.000–4.000 | speaker 1 laughter at 9.410; speaker 2 at 9.410–12.410; speaker 1 at 12.410–13.410 |
| `B5rLHPUJ8qc_22.181-26.888.wav` | speaker 1 at 0.000–0.500; speaker 2 at approximately 0.500–4.000 | speaker 1 at 22.181–22.681; speaker 2 at approximately 22.681–26.181 |
| `jxTaydBnwIg_1.900-11.600.wav` | speaker 1 at 0.000–8.000; background music at 0.000–9.000; speaker 2 starts at 9.000 | speaker 1 at 1.900–9.900; background music at 1.900–10.900; speaker 2 starts at 10.900 |
| `XSC94UbpoRQ_56.000-60.400.wav` | speaker 1 at 0.000–4.000 | speaker 1 at 56.000–60.000 |

For `B5rLHPUJ8qc`, the reviewer's “half a second” and `0:005` descriptions are
recorded as an approximate change at 0.500 s. For `jxTaydBnwIg`, `0:009` is
recorded as 9.000 s.

## Comparison with reference and hypothesis

The compared run uses a 0.25 s scoring collar, includes overlap and evaluates
the full audio. All four recordings have the correct global speaker count
(`reference/prediction = 2/2`), so these cases demonstrate that a correct count
does not imply correct temporal speaker assignment.

### `98EV6H0ZRlU`

- Manual review places the speaker change at source time 12.410 s.
- The raw reference changes near 13.445 s, approximately 1.0 s later.
- The hypothesis uses `SPEAKER_00` on both sides of the manually confirmed
  change in this clip.
- Classification: mixed reference-boundary discrepancy and missed speaker
  change. This clip alone cannot apportion the complete recording's error.

### `B5rLHPUJ8qc`

- Manual review places the change at approximately 22.681 s.
- The raw reference ends the first speaker at 22.867 s and starts the second at
  23.056 s, close to the manual change but with a short gap.
- The hypothesis keeps `SPEAKER_00` across both manually distinguished voices.
- Classification: genuine missed change / cluster identity mixing, with only a
  small reference-boundary offset.

### `jxTaydBnwIg`

- Manual review places the change at 10.900 s.
- The raw reference ends one speaker at 10.725 s and starts the other at
  11.285 s; the manual change lies inside that reference gap.
- The hypothesis keeps `SPEAKER_00` across the two manually distinguished
  voices. Globally, that cluster is mapped to only one reference speaker.
- Classification: strong cluster identity mixing. Background music through
  10.900 s is a candidate embedding contaminant, not a proven cause.

### `XSC94UbpoRQ`

- Manual review finds one unchanged voice from 56.000 to 60.000 s.
- The reference also uses one unchanged speaker label over this interval.
- The hypothesis changes from `SPEAKER_00` to `SPEAKER_01` near 59.700 s even
  though no manual speaker change is present.
- Classification: spurious cluster switch / fragmentation of the same voice.

## Consequence for further analysis

The remaining `SpeakerError` cannot be explained only by an incorrect number
of speakers. The next diagnostic should inspect cached embedding windows and
their assigned cluster IDs around these four intervals. It should compare
within-voice and cross-voice cosine similarity and mark windows that overlap
manual change points or background music.

These partial annotations are evidence for error diagnosis. They are not
complete replacement RTTMs and must not be used to recompute whole-recording
DER without reviewing the remaining audio.
