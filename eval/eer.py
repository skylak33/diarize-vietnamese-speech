"""Reproducible speaker verification EER/minDCF, separate from diarization DER."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def add_protocol_arguments(parser):
    parser.add_argument('--dataset', choices=['vlsp','dolly'], default='vlsp')
    parser.add_argument('--n_pairs', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    parser.add_argument('--manifest', help='CSV: utterance_id,path,speaker_id,split')
    parser.add_argument('--trials', help='Official trials, preferred over sampled pairs')
    parser.add_argument('--trials_format', choices=['auto','vlsp','csv','text'], default='auto',
                        help='vlsp: headerless wav1,wav2,label from Kaggle Public Test; auto: CSV by extension or whitespace text')
    parser.add_argument('--trials_order', choices=['label_first','label_last'], default='label_first')
    parser.add_argument('--audio_root')
    parser.add_argument('--split', choices=['test','public_test','validation','dev'], default='test')
    parser.add_argument('--hf_dataset', default='dolly-vn/dolly-audio-1000h-vietnamese')
    parser.add_argument('--revision', default='main', help='Pin a HF commit for reproducibility')
    parser.add_argument('--max_utterances', type=int, default=10000)
    parser.add_argument('--n_shards', type=int, help='Deprecated: train shards are never evaluated')
    parser.add_argument('--p_target', type=float, default=0.05)
    parser.add_argument('--c_miss', type=float, default=1.0)
    parser.add_argument('--c_fa', type=float, default=1.0)


def evaluate_model(model_path, records, trials, device='auto', p_target=0.05, c_miss=1., c_fa=1.):
    import numpy as np
    from tqdm import tqdm
    from utils.model import load_model
    from utils.audio import decode_audio
    from utils.metrics import verification_metrics
    from diarize.pipeline import extract_embedding
    encoder = load_model(model_path, device)
    embeddings = {}
    for uid in tqdm(sorted({u for _,a,b in trials for u in (a,b)}), desc='Embeddings'):
        audio = decode_audio(records[uid]['audio'])
        embeddings[uid] = extract_embedding(encoder, audio, 0, len(audio))
    scores = [float(np.dot(embeddings[a], embeddings[b])) for _,a,b in trials]
    labels = [label for label,_,_ in trials]
    result = verification_metrics(scores, labels, p_target, c_miss, c_fa)
    result.update(pairs=len(trials), positive_pairs=sum(labels), negative_pairs=len(labels)-sum(labels),
                  model=str(model_path), protocol_sha256=hashlib.sha256(json.dumps(trials).encode()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', help='JSON with results and protocol provenance')
    add_protocol_arguments(parser)
    args = parser.parse_args()
    if args.n_shards is not None:
        parser.error('--n_shards selected training data and is no longer safe; use --manifest')
    from utils.seed import set_seed
    from eval.datasets import load_protocol
    from utils.provenance import environment, sha256
    set_seed(args.seed)
    records, trials, metadata = load_protocol(args)
    result = {**metadata, **evaluate_model(args.model_path, records, trials, args.device,
                                         args.p_target, args.c_miss, args.c_fa)}
    result.update(environment=environment(), model_sha256=sha256(args.model_path))
    for key in ('manifest', 'trials_file'):
        if metadata[key]:
            result[key + '_sha256'] = sha256(metadata[key])
    print(f"Dataset: {result['dataset']}\nProtocol: {result['protocol']}\nPairs: {result['pairs']}\nPositive pairs: {result['positive_pairs']}\nNegative pairs: {result['negative_pairs']}\nEER: {result['EER']:.4f} %\nminDCF: {result['minDCF']:.6f}\nThreshold: {result['threshold']:.6f} (nearest empirical EER operating point)")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
