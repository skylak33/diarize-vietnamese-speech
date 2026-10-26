"""Score every Nth checkpoint in sorted epoch order on exactly the same trials."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.eer import add_protocol_arguments, evaluate_model


def select_checkpoints(model_dir, every=1):
    if every < 1:
        raise ValueError('every must be positive')
    checkpoints = []
    for path in Path(model_dir).glob('model_*.model'):
        match = re.fullmatch(r'model_(\d+)\.model', path.name)
        if match:
            checkpoints.append((int(match[1]), path))
    if not checkpoints:
        raise ValueError(f'No model_NNNN.model checkpoints in {model_dir}')
    return sorted(checkpoints)[::every]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('--output', required=True)
    add_protocol_arguments(parser)
    args = parser.parse_args()
    if args.n_shards is not None:
        parser.error('--n_shards is unsafe; use --manifest')
    from eval.datasets import load_protocol
    from utils.seed import set_seed
    from utils.provenance import environment, sha256
    checkpoints = select_checkpoints(args.model_dir, args.every)
    set_seed(args.seed)
    records, trials, protocol = load_protocol(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    details = dict(arguments=vars(args), protocol=protocol, environment=environment(), results=[])
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch','EER','minDCF'])
        writer.writeheader()
        for epoch, path in checkpoints:
            set_seed(args.seed)
            result = evaluate_model(str(path), records, trials, args.device, args.p_target, args.c_miss, args.c_fa)
            writer.writerow(dict(epoch=epoch, EER=result['EER'], minDCF=result['minDCF']))
            handle.flush()
            details['results'].append(dict(epoch=epoch, model_sha256=sha256(path), **result))
            output.with_suffix('.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
            print(f"epoch {epoch}: EER={result['EER']:.4f}%, minDCF={result['minDCF']:.6f}")


if __name__ == '__main__':
    main()
