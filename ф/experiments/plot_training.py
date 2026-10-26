"""Plot the existing score.txt format without Jupyter."""
import argparse
import csv
import re
from pathlib import Path


def read_training_log(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        epoch = re.search(r'(\d+)\s+epoch', line)
        if not epoch:
            continue
        values = {}
        for field in ('LR','LOSS','ACC'):
            match = re.search(rf'\b{field}\s+([-+\d.eE]+)', line)
            if not match:
                raise ValueError(f'Missing {field} in epoch log line: {line}')
            values[field] = float(match[1])
        rows[int(epoch[1])] = values  # append logs: last occurrence of an epoch wins
    if not rows:
        raise ValueError('No epoch/LR/LOSS/ACC rows found')
    return sorted(rows.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--eer_csv', help='CSV with epoch,EER')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_training_log(args.log)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    def plot(epochs, values, label, filename):
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.plot(epochs, values, marker='o', markersize=3)
        ax.set(xlabel='Epoch', ylabel=label)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
    for key, label, name in (('LOSS','Loss','loss'),('ACC','Accuracy (%)','accuracy'),('LR','Learning rate','lr')):
        plot([e for e,_ in rows], [r[key] for _,r in rows], label, f'training_{name}.png')
    if args.eer_csv:
        with open(args.eer_csv, newline='', encoding='utf-8-sig') as handle:
            eer = sorted((int(r['epoch']),float(r['EER'])) for r in csv.DictReader(handle))
        if not eer:
            raise ValueError('EER CSV has no measurements')
        plot([e for e,_ in eer], [v for _,v in eer], 'EER (%)', 'eer_by_epoch.png')


if __name__ == '__main__':
    main()
