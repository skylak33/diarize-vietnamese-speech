# PLDA backend feasibility

## Decision

PLDA is not implemented or evaluated in this repository. The locally available
ViYT-Diar material is the same 100-recording test corpus used for the reported
DER experiments. Fitting PLDA parameters or calibrating a decision threshold on
those recordings would leak evaluation speaker labels into model selection.

The existing cosine backend therefore remains the only supported scoring
backend. An untrained or randomly initialized PLDA model must not be added as an
inference option.

## Data required before implementation

A valid experiment needs:

- speaker-labeled, in-domain Vietnamese speech for PLDA training;
- enough speakers and sessions to estimate within- and between-speaker
  covariance reliably;
- speaker-disjoint training, development, and held-out test partitions;
- a development set for dimensionality reduction and score/cluster-threshold
  calibration;
- a held-out test set that is not used to select PLDA, AHC, VAD, window, or
  post-merge parameters.

## Minimal implementation plan

1. Extract frozen epoch-34 ECAPA embeddings and record the checkpoint hash,
   embedding dimension, dataset revision, and split manifest.
2. Fit centering and optional LDA only on the PLDA training split, then estimate
   within- and between-speaker covariance on that same split.
3. Serialize a versioned backend artifact containing preprocessing parameters,
   PLDA parameters, training-manifest hash, embedding dimension, and dependency
   versions.
4. Validate artifact dimension and metadata before inference. Keep cosine
   scoring available and make PLDA opt-in.
5. Calibrate clustering thresholds on the development split. Do not use test
   labels or test DER during calibration.
6. Compare cosine and PLDA with identical frozen embeddings and segmentation.
   Report DER, SpeakerError, speaker-count metrics, runtime, and memory on the
   held-out test split.

Implementation is justified only after these inputs exist. Until then, PLDA is
a research hypothesis rather than a repository feature.
