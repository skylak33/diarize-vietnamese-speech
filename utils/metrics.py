import os
import numpy
import torch
from sklearn import metrics


def verification_metrics(scores, labels, p_target=0.05, c_miss=1., c_fa=1.):
    """EER by linear interpolation on tie-grouped ROC; normalized minDCF.

    Accept target when score >= threshold. EER threshold is the nearest
    empirical operating point, since an interpolated EER can require randomization.
    """
    scores, labels = numpy.asarray(scores, dtype=float), numpy.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not numpy.isfinite(scores).all():
        raise ValueError("Scores/labels must be equally sized finite vectors")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("EER requires both binary target and non-target labels")
    if not 0 < p_target < 1 or min(c_miss, c_fa) <= 0:
        raise ValueError("Require 0 < p_target < 1 and positive costs")
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
    thresholds[0] = numpy.nextafter(scores.max(), numpy.inf)
    fnr = 1 - tpr
    diff = fpr - fnr
    hi = int(numpy.flatnonzero(diff >= 0)[0])
    lo = max(0, hi - 1)
    weight = 0 if diff[hi] == diff[lo] else -diff[lo] / (diff[hi] - diff[lo])
    eer = fpr[lo] + weight * (fpr[hi] - fpr[lo])
    costs = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    idx = int(numpy.argmin(costs))
    nearest = int(numpy.argmin(numpy.abs(diff)))
    return dict(EER=float(eer * 100), minDCF=float(costs[idx] / min(c_miss*p_target, c_fa*(1-p_target))),
                threshold=float(thresholds[nearest]), minDCF_threshold=float(thresholds[idx]),
                p_target=p_target, c_miss=c_miss, c_fa=c_fa)


def init_args(args):
    args.score_save_path = os.path.join(args.save_path, "score.txt")
    args.model_save_path = os.path.join(args.save_path, "model")
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    result = verification_metrics(scores, labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    eer = result["EER"]
    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):
    verification_metrics(scores, labels)  # validate binary classes and finite scores
    fprs, tprs, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
    thresholds[0] = numpy.nextafter(max(scores), numpy.inf)
    return (1 - tprs).tolist(), fprs.tolist(), thresholds.tolist()


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
