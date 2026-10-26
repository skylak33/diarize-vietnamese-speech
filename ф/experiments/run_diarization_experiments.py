"""Controlled one-factor-at-a-time diarization ablations (not a Cartesian grid)."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def configurations(args):
    base = dict(method=args.method, ahc_threshold=args.threshold,
                vad_threshold=args.vad_threshold, window_sec=args.window_sec,
                step_sec=args.step_sec or args.window_sec / 2, speaker_mode=args.speaker_mode)
    variations = {
        'clustering': ('method', ['ahc', 'spectral']),
        'ahc_threshold': ('ahc_threshold', [0.45,0.50,0.55,0.60,0.65,0.70,0.75]),
        'vad_threshold': ('vad_threshold', [0.3,0.4,0.5,0.6,0.7]),
        'window': ('window_sec', [1.,1.5,2.]),
        'speaker_count': ('speaker_mode', ['known','automatic']),
    }
    selected = variations if args.experiment == 'all' else [args.experiment]
    configs = []
    for experiment in selected:
        key, values = variations[experiment]
        for value in values:
            config = {**base, key:value}
            if experiment == 'ahc_threshold':
                config.update(method='ahc', speaker_mode='automatic')
            if key == 'window_sec' and args.step_sec is None:
                config['step_sec'] = value / 2
            if not 0 < config['step_sec'] <= config['window_sec']:
                raise ValueError('Require 0 < step_sec <= window_sec in every configuration')
            if config not in configs:
                configs.append(config)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('audio_dir','ref_rttm_dir','model','output_dir'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--experiment', choices=['clustering','ahc_threshold','vad_threshold','window','speaker_count','all'], default='clustering')
    parser.add_argument('--method', choices=['ahc','spectral'], default='ahc')
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--vad_threshold', type=float, default=0.5)
    parser.add_argument('--window_sec', type=float, default=1.5)
    parser.add_argument('--step_sec', type=float)
    parser.add_argument('--speaker_mode', choices=['known','automatic'], default='automatic')
    parser.add_argument('--max_speakers', type=int, default=10)
    parser.add_argument('--collar', type=float, default=0.25)
    parser.add_argument('--skip_overlap', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--dry_run', action='store_true', help='Print configurations without loading audio/model')
    parser.add_argument('--max_configs', type=int, default=32, help='Abort if plan exceeds limit')
    args = parser.parse_args()
    configs = configurations(args)
    if len(configs) > args.max_configs:
        parser.error(f'{len(configs)} configurations exceed --max_configs={args.max_configs}')
    print(json.dumps(configs, indent=2))
    print(f'Configurations: {len(configs)}')
    if args.dry_run:
        return
    from diarize.pipeline import diarize
    from utils.model import load_model
    from utils.seed import set_seed
    from utils.provenance import sha256, environment
    from eval.der import load_rttm, parse_rttm, speaker_count, evaluate_annotations, save_results
    import soundfile as sf
    refs = load_rttm(args.ref_rttm_dir)
    audios = {}
    for path in sorted(Path(args.audio_dir).rglob('*')):
        if path.suffix.lower() not in {'.wav','.flac','.ogg'}:
            continue
        if path.stem in audios:
            raise ValueError(f'Duplicate audio recording ID: {path.stem}')
        audios[path.stem] = path
    if refs.keys() != audios.keys():
        raise ValueError(f'Audio/reference IDs differ; missing audio={refs.keys()-audios.keys()}, missing reference={audios.keys()-refs.keys()}')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'experiment_manifest.json').exists():
        raise FileExistsError('Output already contains an experiment; choose a new --output_dir')
    metadata = dict(arguments=vars(args), configurations=configs, environment=environment(),
                    model_sha256=sha256(args.model),
                    audio_sha256={k:sha256(v) for k,v in audios.items()},
                    reference_sha256={str(p):sha256(p) for p in sorted(Path(args.ref_rttm_dir).rglob('*.rttm'))})
    (out / 'experiment_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    durations = {key:sf.info(path).duration for key,path in audios.items()}
    set_seed(args.seed)
    encoder = load_model(args.model, args.device)
    fields = ['method','ahc_threshold','vad_threshold','window_sec','step_sec','speaker_mode',
              'DER','Miss','FalseAlarm','SpeakerError','SpeakerCountAccuracy','SpeakerCountMAE','config_id']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, config in enumerate(configs):
            config_id = f'config_{index:03d}'
            folder = out / config_id
            folder.mkdir()
            (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            hypotheses = {}
            for file_id, audio in audios.items():
                n = speaker_count(refs[file_id]) if config['speaker_mode'] == 'known' else None
                # Silence has zero reference speakers; an empty hypothesis is the oracle-count case.
                rttm = folder / 'rttm' / (file_id + '.rttm')
                if n == 0:
                    rttm.parent.mkdir(exist_ok=True)
                    rttm.write_text('', encoding='utf-8')
                else:
                    diarize(str(audio), args.model, n_speakers=n, method=config['method'],
                            threshold=config['ahc_threshold'], vad_threshold=config['vad_threshold'],
                            window_sec=config['window_sec'], step_sec=config['step_sec'],
                            output_rttm=str(rttm), seed=args.seed, device=args.device,
                            max_speakers=args.max_speakers, encoder=encoder)
                hypotheses[file_id] = parse_rttm(str(rttm)).get(file_id, [])
            result = evaluate_annotations(refs, hypotheses, args.collar, args.skip_overlap, durations)
            save_results(result, folder / 'der.json', folder / 'per_file.csv')
            total = result['__total__']
            writer.writerow({**config, 'config_id':config_id, 'DER':total['DER'],
                'Miss':total['miss'], 'FalseAlarm':total['false_alarm'], 'SpeakerError':total['speaker_error'],
                'SpeakerCountAccuracy':total['speaker_count_accuracy'], 'SpeakerCountMAE':total['speaker_count_mae']})
            handle.flush()
            print(f"{config_id}: DER={total['DER']}")


if __name__ == '__main__':
    main()
