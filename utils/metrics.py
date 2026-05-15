import os
import numpy
import torch
from sklearn import metrics
from operator import itemgetter


def init_args(args):
    args.score_save_path = os.path.join(args.save_path, "score.txt")
    args.model_save_path = os.path.join(args.save_path, "model")
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100
    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=itemgetter(1),
        )
    )
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm
    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


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
