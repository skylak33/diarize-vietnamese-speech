"""
evalDER — Подсчёт DER (Diarization Error Rate).

DER = Miss + False Alarm + Speaker Error
"""

import os
import sys
import argparse
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_rttm(rttm_path: str) -> dict[str, list[tuple[float, float, str]]]:
    result = {}
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            file_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            result.setdefault(file_id, []).append((start, start + duration, speaker_id))
    return result


def compute_der_pyannote(
    hyp_rttm: str,
    ref_rttm: str,
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> dict:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def rttm_to_annotation(rttm_path: str) -> dict[str, Annotation]:
        annotations = {}
        for file_id, segs in parse_rttm(rttm_path).items():
            ann = Annotation(uri=file_id)
            for start, end, spk in segs:
                ann[Segment(start, end)] = spk
            annotations[file_id] = ann
        return annotations

    ref_annotations = rttm_to_annotation(ref_rttm)
    hyp_annotations = rttm_to_annotation(hyp_rttm)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)

    results = {}
    total_der = 0.0
    n_files = 0

    for file_id in ref_annotations:
        if file_id not in hyp_annotations:
            print(f"[der] Предупреждение: {file_id} отсутствует в гипотезе")
            continue
        components = metric(ref_annotations[file_id], hyp_annotations[file_id], detailed=True)
        der = abs(metric)
        total = components["total"]
        results[file_id] = {
            "DER":           der * 100,
            "miss":          components["missed detection"] / total * 100,
            "false_alarm":   components["false alarm"]      / total * 100,
            "speaker_error": components["confusion"]        / total * 100,
        }
        total_der += der
        n_files += 1
        metric.reset()

    if n_files == 0:
        return {}
    results["__total__"] = {"DER": total_der / n_files * 100, "n_files": n_files}
    return results


def compute_der_simple(hyp_rttm: str, ref_rttm: str, collar: float = 0.25) -> dict:
    import numpy
    from scipy.optimize import linear_sum_assignment

    ref_data = parse_rttm(ref_rttm)
    hyp_data = parse_rttm(hyp_rttm)
    total_ref_dur = total_miss = total_fa = total_spk_err = 0.0

    for file_id in ref_data:
        ref_segs = ref_data[file_id]
        hyp_segs = hyp_data.get(file_id, [])
        ref_speakers = sorted(set(s for _, _, s in ref_segs))
        hyp_speakers = sorted(set(s for _, _, s in hyp_segs))
        if not ref_speakers or not hyp_speakers:
            continue

        def overlap(seg1, seg2):
            return max(0.0, min(seg1[1], seg2[1]) + collar - max(seg1[0], seg2[0]) + collar)

        n_ref, n_hyp = len(ref_speakers), len(hyp_speakers)
        overlap_matrix = numpy.zeros((n_ref, n_hyp))
        for i, rs in enumerate(ref_speakers):
            rs_segs = [(s, e) for s, e, sp in ref_segs if sp == rs]
            for j, hs in enumerate(hyp_speakers):
                hs_segs = [(s, e) for s, e, sp in hyp_segs if sp == hs]
                overlap_matrix[i, j] = sum(
                    overlap(r, h) for r in rs_segs for h in hs_segs
                )

        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        matched_dur = overlap_matrix[row_ind, col_ind].sum()
        ref_total = sum(e - s for s, e, _ in ref_segs)
        hyp_total = sum(e - s for s, e, _ in hyp_segs)

        total_ref_dur += ref_total
        total_miss += max(0.0, ref_total - matched_dur)
        total_fa += max(0.0, hyp_total - matched_dur)
        total_spk_err += max(0.0, ref_total - matched_dur - max(0.0, ref_total - matched_dur))

    if total_ref_dur == 0:
        return {"DER": 0.0, "miss": 0.0, "false_alarm": 0.0, "speaker_error": 0.0}

    return {
        "DER":           (total_miss + total_fa + total_spk_err) / total_ref_dur * 100,
        "miss":          total_miss    / total_ref_dur * 100,
        "false_alarm":   total_fa      / total_ref_dur * 100,
        "speaker_error": total_spk_err / total_ref_dur * 100,
    }


def print_results(results: dict) -> None:
    total = results.get("__total__")
    file_results = {k: v for k, v in results.items() if k != "__total__"}
    if file_results:
        print(f"\n{'Файл':<30} {'DER':>7} {'Miss':>7} {'FA':>7} {'SpkErr':>8}")
        print("-" * 62)
        for file_id, r in sorted(file_results.items()):
            print(f"{file_id:<30} {r['DER']:>6.2f}% {r['miss']:>6.2f}% "
                  f"{r['false_alarm']:>6.2f}% {r['speaker_error']:>7.2f}%")
    if total:
        print("=" * 62)
        print(f"\n  Файлов оценено : {total['n_files']}")
        print(f"  DER (среднее)  : {total['DER']:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarization Error Rate (DER)")
    parser.add_argument("--hyp",         type=str)
    parser.add_argument("--ref",         type=str)
    parser.add_argument("--hyp_dir",     type=str)
    parser.add_argument("--ref_dir",     type=str)
    parser.add_argument("--audio",       type=str)
    parser.add_argument("--model",       type=str)
    parser.add_argument("--n_speakers",  type=int, default=None)
    parser.add_argument("--method",      type=str, default="ahc")
    parser.add_argument("--collar",      type=float, default=0.25)
    parser.add_argument("--simple",      action="store_true")
    args = parser.parse_args()

    if args.audio and args.model and args.ref:
        import tempfile
        from diarize.pipeline import diarize
        print("[der] Запускаем диаризацию...")
        rttm_str = diarize(audio_path=args.audio, model_path=args.model,
                           n_speakers=args.n_speakers, method=args.method)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as f:
            f.write(rttm_str + "\n")
            hyp_tmp = f.name
        print("[der] Считаем DER...")
        r = (compute_der_simple if args.simple else
             lambda h, ref, **kw: compute_der_pyannote(h, ref, **kw))(
            hyp_tmp, args.ref, collar=args.collar)
        print_results(r if isinstance(r, dict) and "__total__" in r
                      else {**r, "__total__": {"DER": r["DER"], "n_files": 1}})
        os.unlink(hyp_tmp)

    elif args.hyp and args.ref:
        if args.simple:
            r = compute_der_simple(args.hyp, args.ref, collar=args.collar)
            print(f"\n  DER          : {r['DER']:.2f}%")
            print(f"  Miss         : {r['miss']:.2f}%")
            print(f"  False Alarm  : {r['false_alarm']:.2f}%")
            print(f"  Speaker Error: {r['speaker_error']:.2f}%")
        else:
            print_results(compute_der_pyannote(args.hyp, args.ref, collar=args.collar))

    elif args.hyp_dir and args.ref_dir:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as ch, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as cr:
            for ref_path in sorted(glob.glob(os.path.join(args.ref_dir, "*.rttm"))):
                file_id = os.path.splitext(os.path.basename(ref_path))[0]
                hyp_path = os.path.join(args.hyp_dir, file_id + ".rttm")
                if os.path.exists(hyp_path):
                    ch.write(open(hyp_path).read() + "\n")
                    cr.write(open(ref_path).read() + "\n")
        print_results(compute_der_pyannote(ch.name, cr.name, collar=args.collar))
        os.unlink(ch.name)
        os.unlink(cr.name)
    else:
        parser.print_help()
