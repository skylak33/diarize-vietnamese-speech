# ViYT-Diar manual reference audit — 2026-11-02

These three RTTM files preserve reviewer-validated speaker turns. The reviewer
confirmed that the supplied boundaries are sufficiently accurate for subsequent
local evaluation. They do not replace the published ViYT-Diar references, so
results using them must be identified as using the `manual-audit reference`.

## Interpretation

- Music is background audio, not a speaker.
- Explicit silence is excluded from the RTTM.
- `C4HFERLpRfI`: speaker 1 at 0–11 s; speaker 2 at 12–43 s; speaker 1 laughter at
  43–45 s; speaker 2 at 45–62 s.
- `Uaq5N3FbQb4`: speaker 1 at 0–36 s over music; speaker 2 at 36–60 s with quieter
  background music.
- `ed0Vt0NxJUI`: speaker 1 at 0–8 s; speaker 2 at 8–57 s; silence at 57–62 s.

## Evaluation status

These annotations are the accepted project reference for the three audited
recordings. Their timestamps are expressed to the nearest second, matching the
manual review. They may be used for local DER analysis as long as results remain
separate from scores computed against the published ViYT-Diar reference.

## SHA256

| File | SHA256 |
|---|---|
| `C4HFERLpRfI.rttm` | `439fe5c686a9eadbc5466a863908173af682c0dd91b958e8bcbb4359aeee6ff0` |
| `Uaq5N3FbQb4.rttm` | `23fa882d0109f9b9dfc8a6af400a6f35276b2449369cd138d37bc107deda7db0` |
| `ed0Vt0NxJUI.rttm` | `6d0b40af72276b5d996a1a7bb4bc985902ccbb1d3089a55a38e5206215396440` |
