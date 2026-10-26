"""Explicit verification protocols: never relabel a training split as test.

Local manifest schema is project-defined, not a claimed VLSP distribution layout:
CSV: utterance_id,path,speaker_id,split. Trials: label path1 path2 or CSV label,utterance1,utterance2.
"""
import csv
import os
import random
from pathlib import Path
from collections import defaultdict

EVAL_SPLITS = {'test', 'public_test', 'validation', 'dev'}


def read_manifest(path, audio_root=None, split='test'):
    if split not in EVAL_SPLITS:
        raise ValueError('Evaluation requires test/public_test/validation/dev, never train')
    root = Path(audio_root or Path(path).parent)
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'utterance_id','path','speaker_id','split'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest requires utterance_id,path,speaker_id,split columns')
        rows = list(reader)
    if split in {'dev', 'validation'} and any(r['split'] in {'test','public_test'} for r in rows):
        raise ValueError('Manifest contains official test data; select its test split for evaluation')
    selected = {}
    train_speakers = {r['speaker_id'] for r in rows if r['split'] == 'train'}
    train_paths = {(root / r['path']).resolve() for r in rows if r['split'] == 'train'}
    for row in rows:
        if row['split'] != split:
            continue
        uid = row['utterance_id']
        audio = (root / row['path']).resolve()
        if not uid or not row['speaker_id'] or uid in selected:
            raise ValueError('Empty/duplicate utterance ID or empty speaker ID')
        if audio in train_paths or row['speaker_id'] in train_speakers:
            raise ValueError(f'Train/evaluation leakage in manifest: {uid}')
        if not audio.is_file():
            raise FileNotFoundError(audio)
        selected[uid] = {'audio': str(audio), 'speaker': row['speaker_id']}
    if not selected:
        raise ValueError(f'No records for split {split} in {path}')
    return selected


def read_vlsp_trial_rows(text: str, path) -> list[tuple[str, str, str]]:
    """Kaggle notebook cell 14: headerless wav1,wav2,label (including .txt files)."""
    rows = []
    for number, fields in enumerate(csv.reader(text.splitlines(), strict=True), 1):
        if not fields or (len(fields) == 1 and not fields[0].strip()):
            continue
        fields = [field.strip() for field in fields]
        if len(fields) != 3 or not all(fields) or fields[2] not in {'0', '1'}:
            raise ValueError(f'{path}:{number}: expected wav1,wav2,label with binary label and no header')
        a, b, label = fields
        rows.append((label, a, b))
    return rows


def read_trials(path, records=None, audio_root=None, order='label_first', trials_format='auto'):
    text = Path(path).read_text(encoding='utf-8-sig')
    if trials_format not in {'auto', 'vlsp', 'csv', 'text'}:
        raise ValueError('Unknown trials_format')
    if trials_format == 'vlsp':
        rows = read_vlsp_trial_rows(text, path)
    elif trials_format == 'csv' or (trials_format == 'auto' and Path(path).suffix.lower() == '.csv'):
        reader = csv.DictReader(text.splitlines())
        if not {'label','utterance1','utterance2'} <= set(reader.fieldnames or []):
            raise ValueError('Trials CSV requires label,utterance1,utterance2')
        rows = [(r['label'], r['utterance1'], r['utterance2']) for r in reader]
    else:
        rows = []
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError('Trials must contain exactly 3 fields; use explicit CSV conversion')
            rows.append(parts if order == 'label_first' else (parts[2], parts[0], parts[1]))
    labels = {'0':0, '1':1, 'target':1, 'nontarget':0}
    trials = []
    resolve_directly = records is None
    records = {} if resolve_directly else records.copy()
    for label, a, b in rows:
        if label not in labels or a == b:
            raise ValueError(f'Invalid trial: {label} {a} {b}')
        for uid in (a,b):
            if resolve_directly:
                if not audio_root:
                    raise ValueError('Trials without manifest require --audio_root')
                audio = (Path(audio_root) / uid).resolve()
                if not audio.is_file():
                    raise FileNotFoundError(audio)
                records[uid] = {'audio':str(audio), 'speaker':None}
            elif uid not in records:
                raise ValueError(f'Trial refers outside selected evaluation split: {uid}')
        if records[a]['speaker'] is not None and labels[label] != int(records[a]['speaker'] == records[b]['speaker']):
            raise ValueError(f'Trial label contradicts speaker metadata: {a}, {b}')
        trials.append((labels[label], a, b))
    if not trials or {t[0] for t in trials} != {0,1}:
        raise ValueError('Protocol needs both positive and negative trials')
    return records, trials


def generate_pairs(records, n_pairs=20000, seed=42):
    """Balanced sampling with replacement, speaker-uniform. No self-positive pairs."""
    if n_pairs < 2:
        raise ValueError('n_pairs must be >= 2')
    groups = defaultdict(list)
    for uid, row in sorted(records.items()):
        groups[row['speaker']].append(uid)
    speakers = sorted(groups)
    positives = [s for s in speakers if len(groups[s]) >= 2]
    if len(speakers) < 2 or not positives:
        raise ValueError('Need >=2 speakers and >=2 utterances for at least one speaker')
    rng = random.Random(seed)
    trials = []
    for _ in range(n_pairs // 2):
        a,b = rng.sample(groups[rng.choice(positives)], 2)
        trials.append((1,a,b))
    for _ in range(n_pairs - len(trials)):
        a,b = rng.sample(speakers, 2)
        trials.append((0,rng.choice(groups[a]),rng.choice(groups[b])))
    rng.shuffle(trials)
    return trials


def load_protocol(args):
    manifest = args.manifest or os.environ.get(args.dataset.upper() + '_EVAL_MANIFEST')
    trials_path = args.trials or os.environ.get(args.dataset.upper() + '_TRIALS')
    audio_root = args.audio_root or os.environ.get(args.dataset.upper() + '_AUDIO_ROOT')
    actual_split = args.split
    resolved_revision = args.revision
    if args.max_utterances < 2:
        raise ValueError('max_utterances must be >= 2')
    if manifest:
        records = read_manifest(manifest, audio_root, args.split)
    elif args.dataset == 'vlsp':
        if not trials_path:
            raise ValueError('VLSP Public Test is not bundled. Set --trials and --audio_root, or --manifest. No training fallback.')
        records = None
    elif trials_path:
        records = None
    else:
        from datasets import load_dataset, get_dataset_split_names, Audio
        from huggingface_hub import HfApi
        resolved_revision = HfApi().dataset_info(args.hf_dataset, revision=args.revision).sha
        splits = get_dataset_split_names(args.hf_dataset, revision=resolved_revision)
        official = next((s for s in ('test','public_test') if s in splits), None)
        split = official or args.split
        actual_split = split
        if split not in splits or split not in EVAL_SPLITS:
            raise ValueError(f'{args.hf_dataset} exposes {splits}; no safe evaluation split. Supply a held-out --manifest and exclude its speakers from training.')
        ds = load_dataset(args.hf_dataset, split=split, streaming=True, revision=resolved_revision)
        ds = ds.cast_column('audio', Audio(decode=False))
        records = {}
        for i, row in enumerate(ds):
            if i >= args.max_utterances:
                break
            speaker = row.get('voice_id', row.get('speaker_id'))
            if speaker is None:
                raise ValueError('Dolly record has no voice_id/speaker_id')
            records[str(i)] = {'audio':row['audio'], 'speaker':str(speaker)}
    if trials_path:
        records, trials = read_trials(trials_path, records, audio_root, args.trials_order, args.trials_format)
        protocol = 'provided trials (complete; n_pairs ignored)'
    else:
        trials = generate_pairs(records, args.n_pairs, args.seed)
        protocol = 'speaker-uniform balanced sampling with replacement'
    return records, trials, {'dataset': 'VLSP 2021' if args.dataset == 'vlsp' else 'Dolly Vietnamese',
        'protocol':protocol, 'manifest':manifest, 'trials_file':trials_path,
        'trials_format':args.trials_format if trials_path else None,
        'split':actual_split, 'seed':args.seed, 'hf_dataset':args.hf_dataset if not manifest and args.dataset == 'dolly' else None,
        'revision':resolved_revision}
