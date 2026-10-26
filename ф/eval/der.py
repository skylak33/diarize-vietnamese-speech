"""Corpus DER using pyannote.metrics. Rates are percentages; components are seconds."""
import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_rttm(rttm_path: str) -> dict:
    result, channels = {}, {}
    for number, line in enumerate(Path(rttm_path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith((';', '#')):
            continue
        parts = line.split()
        if parts[0] != 'SPEAKER':
            continue
        if len(parts) < 9:
            raise ValueError(f'{rttm_path}:{number}: malformed SPEAKER row')
        start, duration = float(parts[3]), float(parts[4])
        if not math.isfinite(start + duration) or start < 0 or duration <= 0:
            raise ValueError(f'{rttm_path}:{number}: invalid time/duration')
        channels.setdefault(parts[1], set()).add(parts[2])
        if len(channels[parts[1]]) > 1:
            raise ValueError('Evaluate RTTM channels separately')
        result.setdefault(parts[1], []).append((start, start + duration, parts[7]))
    return {key: sorted(value) for key, value in result.items()}


def load_rttm(path: str, allow_empty=False) -> dict:
    path = Path(path)
    if not path.is_dir():
        data = parse_rttm(str(path))
        return data if data or allow_empty else {path.stem: []}
    files = sorted(path.rglob('*.rttm'))
    if not files and not allow_empty:
        raise ValueError(f'No RTTM files in {path}')
    result = {}
    for file in files:
        data = parse_rttm(str(file)) or {file.stem: []}
        if result.keys() & data.keys():
            raise ValueError(f'Duplicate recording IDs in {file}')
        result.update(data)
    return result


def speaker_count(segments) -> int:
    return len({speaker for _, _, speaker in segments})


def _rates(row: dict) -> dict:
    total = row['duration']
    for name in ('miss', 'false_alarm', 'speaker_error'):
        row[name] = 100 * row[name + '_seconds'] / total if total else None
    row['DER'] = sum(row[k] for k in ('miss', 'false_alarm', 'speaker_error')) if total else None
    return row


def aggregate_results(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('No reference recordings to evaluate')
    total = _rates({key: sum(row[key] for row in rows) for key in
                    ('duration', 'miss_seconds', 'false_alarm_seconds', 'speaker_error_seconds')})
    total.update(n_files=len(rows),
                 speaker_count_accuracy=sum(r['exact_match'] for r in rows) / len(rows),
                 speaker_count_mae=sum(r['abs_error'] for r in rows) / len(rows))
    return total


def evaluate_annotations(ref_data: dict, hyp_data: dict, collar=0.25,
                         skip_overlap=False, durations=None) -> dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    if collar < 0:
        raise ValueError('collar must be non-negative')
    extra = hyp_data.keys() - ref_data.keys()
    if extra:
        raise ValueError(f'Hypothesis IDs absent from reference: {sorted(extra)}')
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    results = {}
    for file_id, ref_segments in sorted(ref_data.items()):
        hyp_segments = hyp_data.get(file_id, [])
        def annotation(segments):
            ann = Annotation(uri=file_id)
            for track, (start, end, speaker) in enumerate(segments):
                ann[Segment(start, end), track] = speaker
            return ann
        end = (durations[file_id] if durations is not None else
               max((e for _, e, _ in ref_segments + hyp_segments), default=0))
        if any(e > end + 0.002 for _, e, _ in ref_segments + hyp_segments):
            raise ValueError(f'{file_id}: RTTM exceeds recording duration')
        uem = Timeline([Segment(0, end)]) if end > 0 else Timeline()
        c = metric(annotation(ref_segments), annotation(hyp_segments), uem=uem, detailed=True)
        nr, nh = speaker_count(ref_segments), speaker_count(hyp_segments)
        results[file_id] = _rates(dict(duration=float(c['total']),
            miss_seconds=float(c['missed detection']), false_alarm_seconds=float(c['false alarm']),
            speaker_error_seconds=float(c['confusion']), speakers_ref=nr, speakers_hyp=nh,
            N_ref=nr, N_pred=nh, abs_error=abs(nr-nh), exact_match=nr == nh))
    results['__total__'] = aggregate_results(list(results.values()))
    return results


def compute_der_pyannote(hyp_rttm: str, ref_rttm: str, collar=0.25, skip_overlap=False) -> dict:
    return evaluate_annotations(load_rttm(ref_rttm), load_rttm(hyp_rttm, allow_empty=True), collar, skip_overlap)


def save_results(results, output=None, csv_path=None):
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for row in results.values() for k in row))
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['file'] + keys)
            writer.writeheader()
            writer.writerows(dict(file=name, **row) for name, row in results.items())


def print_results(results):
    def fmt(value):
        return 'undefined' if value is None else f'{value:.3f}'
    print('Per-file:\nfile | duration | speakers_ref | speakers_hyp | DER | Miss | FA | SpkErr')
    for file, row in results.items():
        if file != '__total__':
            print(file, *[fmt(row[k]) for k in ('duration','N_ref','N_pred','DER','miss','false_alarm','speaker_error')], sep=' | ')
    total = results['__total__']
    print(f"\nCorpus:\nFiles: {total['n_files']}\nReference speech duration: {total['duration']:.3f} s")
    for label, key in (('DER','DER'),('Miss','miss'),('False Alarm','false_alarm'),('Speaker Error','speaker_error')):
        print(f'{label}: {fmt(total[key])} %')
    print(f"Speaker Count Accuracy: {total['speaker_count_accuracy']:.4f}\nMAE speaker count: {total['speaker_count_mae']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('hyp','ref','hyp_dir','ref_dir','audio','model','output','csv'):
        parser.add_argument('--' + flag)
    parser.add_argument('--collar', type=float, default=0.25)
    parser.add_argument('--skip_overlap', action='store_true')
    parser.add_argument('--n_speakers', type=int)
    parser.add_argument('--method', choices=['ahc','spectral'], default='ahc')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.audio:
        if not args.model or not args.ref:
            parser.error('--audio requires --model and --ref')
        from diarize.pipeline import diarize
        with tempfile.TemporaryDirectory() as tmp:
            hyp = str(Path(tmp) / (Path(args.audio).stem + '.rttm'))
            diarize(args.audio, args.model, n_speakers=args.n_speakers, method=args.method,
                    output_rttm=hyp, device=args.device, seed=args.seed)
            results = compute_der_pyannote(hyp, args.ref, args.collar, args.skip_overlap)
    else:
        if not ((args.hyp and args.ref) or (args.hyp_dir and args.ref_dir)):
            parser.error('Provide --hyp/--ref or --hyp_dir/--ref_dir')
        results = compute_der_pyannote(args.hyp or args.hyp_dir, args.ref or args.ref_dir,
                                       args.collar, args.skip_overlap)
    print_results(results)
    save_results(results, args.output, args.csv)


if __name__ == '__main__':
    main()
