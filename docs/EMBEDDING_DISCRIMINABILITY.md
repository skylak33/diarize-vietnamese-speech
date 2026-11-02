# ECAPA embedding discriminability study

## Status

The requested same-speaker/different-speaker study was not executed. No
speaker-labeled development split that is separate from the existing ViYT-Diar
test evaluation is available locally. Test annotations must not be used to
decide whether or how to fine-tune ECAPA.

The epoch-34 encoder remains frozen. Current diarization ablations can identify
clustering and speaker-count errors, but they cannot establish that the encoder
itself needs additional training.

## Reproducible analysis protocol

When a speaker-disjoint development split is available:

1. Extract embeddings with the production window and VAD settings. Record the
   checkpoint hash and the complete effective configuration.
2. Assign a reference speaker to a window only when the window contains one
   unambiguous active speaker. Exclude silence, unlabeled regions, and windows
   containing multiple reference speakers from this diagnostic.
3. L2-normalize embeddings and compute same-speaker pairs across distinct
   utterances or sessions. Compute different-speaker pairs with deterministic,
   speaker-balanced sampling so prolific speakers do not dominate.
4. Save pair counts and summary statistics (mean, standard deviation, median,
   and useful quantiles) for both cosine-similarity distributions.
5. Generate normalized histograms and report ROC AUC, EER, and a threshold
   selected on development data. Break down within-speaker variability by
   speaker and, where metadata permits, by session and acoustic condition.
6. Repeat only the necessary comparison for alternative window lengths; do not
   reuse embeddings when segmentation changes.

## Decision rule for fine-tuning

Fine-tuning should be considered only if development-set distributions overlap
substantially after threshold, VAD, window, and clustering alternatives have
been evaluated, and if a speaker-disjoint Vietnamese training corpus is
available. Any adapted model must be compared with the frozen epoch-34 model on
an untouched test split using both verification diagnostics and end-to-end DER.

Until that evidence exists, ECAPA fine-tuning is not justified and no training
experiment is recommended from the current test-only results.
