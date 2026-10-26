import argparse
import sys
import warnings

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import init_args
from utils.seed import set_seed
import json
from train.trainer import ECAPAModel

parser = argparse.ArgumentParser(description="ECAPA-TDNN trainer")

parser.add_argument("--num_frames",    type=int,   default=200)
parser.add_argument("--max_epoch",     type=int,   default=80)
parser.add_argument("--batch_size",    type=int,   default=400)
parser.add_argument("--n_cpu",         type=int,   default=4)
parser.add_argument("--test_step",     type=int,   default=1)
parser.add_argument("--lr",            type=float, default=0.001)
parser.add_argument("--lr_decay",      type=float, default=0.97)
parser.add_argument("--save_path",     type=str,   default="exps/exp1")
parser.add_argument("--initial_model", type=str,   default="")
parser.add_argument("--C",             type=int,   default=1024)
parser.add_argument("--m",             type=float, default=0.2)
parser.add_argument("--s",             type=float, default=30)
parser.add_argument("--n_class",       type=int,   default=5994)
parser.add_argument("--max_shards",    type=int,   default=None)
parser.add_argument("--musan_path",    type=str,   default="")
parser.add_argument("--rir_path",      type=str,   default="")
parser.add_argument("--eval",          dest="eval", action="store_true")

parser.add_argument('--dataset', choices=['dolly','vlsp'], default='dolly')
parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--augmentation', choices=['none','noise','rir','full'], default=None,
                    help='Default: infer waveform augmentation from supplied paths; keep legacy SpecAugment')
parser.add_argument('--metadata_path')
parser.add_argument('--audio_base')
parser.add_argument('--exclude_speakers', help='Text file: one held-out speaker ID per line')


def main():
    args = parser.parse_args()
    if min(args.num_frames, args.max_epoch, args.batch_size, args.test_step) <= 0:
        parser.error('Frame, epoch, batch and checkpoint intervals must be positive')
    set_seed(args.seed)
    args = init_args(args)
    from utils.provenance import environment, sha256
    config = {**vars(args), 'environment': environment()}
    for key in ('metadata_path', 'exclude_speakers', 'initial_model'):
        if getattr(args, key):
            config[key + '_sha256'] = sha256(getattr(args, key))
    Path(args.save_path, 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if args.n_cpu != 0:
        warnings.warn('Iterable training loaders use num_workers=0 for deterministic labels/order; --n_cpu is retained for compatibility')
    if args.eval:
        model = ECAPAModel(**vars(args))
        model.load_parameters(args.initial_model)
        print('Model loaded; use eval/eer.py for verification evaluation')
        return
    if args.dataset == 'vlsp':
        from train.loader_vlsp import VLSPLoader as TrainLoader
    else:
        from train.loader_hf import TrainLoader
    loader = TrainLoader(**vars(args))
    batches = torch.utils.data.DataLoader(loader, batch_size=args.batch_size,
                                         num_workers=0, drop_last=True)
    model = ECAPAModel(**vars(args))
    checkpoints = sorted(Path(args.model_save_path).glob('model_*.model'))
    epoch = 1
    if args.initial_model:
        # Transfer learning: never carry a classifier whose speaker labels are unknown.
        model.load_parameters(args.initial_model, encoder_only=True)
    elif checkpoints:
        checkpoint = checkpoints[-1]
        if args.dataset == 'dolly':
            mapping_path = Path(str(checkpoint) + '.speakers.json')
            if not mapping_path.exists():
                raise ValueError('Legacy Dolly checkpoint has no speaker mapping. Use --initial_model for encoder-only transfer into a new --save_path')
            loader.speaker_map = json.loads(mapping_path.read_text())
            loader.speaker_counter = len(loader.speaker_map)
        model.load_parameters(checkpoint)
        epoch = int(checkpoint.stem.split('_')[-1]) + 1
        warnings.warn('Weight-only continuation: optimizer state is reset; record this as a new training stage')
    # Save label mapping alongside every checkpoint, including intermediate batches.
    model.training_loader = loader
    with open(args.score_save_path, 'a', encoding='utf-8') as score:
        while epoch <= args.max_epoch:
            set_seed(args.seed + epoch - 1)
            loss, lr, acc = model.train_network(epoch, batches, args.model_save_path)
            if epoch % args.test_step == 0:
                model.save_parameters(str(Path(args.model_save_path) / f'model_{epoch:04d}.model'))
            score.write(f'{epoch} epoch, LR {lr:.10g}, LOSS {loss:.10g}, ACC {acc:.6f}%\n')
            score.flush()
            epoch += 1


if __name__ == '__main__':
    main()
