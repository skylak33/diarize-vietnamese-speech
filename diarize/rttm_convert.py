"""
rttm_convert.py — конвертация между RTTM и Audacity Labels.
"""

import sys
import os
import argparse


def rttm_to_audacity(rttm_path: str, out_path: str) -> None:
    lines = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            start    = float(parts[3])
            duration = float(parts[4])
            end      = start + duration
            speaker  = parts[7]
            lines.append(f"{start:.3f}\t{end:.3f}\t{speaker}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Записано {len(lines)} меток → {out_path}")
    print("Открой в Audacity: File → Import → Labels")


def audacity_to_rttm(labels_path: str, out_path: str, file_id: str) -> None:
    lines = []
    with open(labels_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"  Строка {i} пропущена (формат): {line!r}")
                continue
            start    = float(parts[0])
            end      = float(parts[1])
            duration = end - start
            speaker  = parts[2].strip()
            if duration <= 0:
                print(f"  Строка {i} пропущена (duration={duration:.3f})")
                continue
            lines.append(
                f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
            )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Записано {len(lines)} сегментов → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("to_audacity", help="RTTM → Audacity labels")
    p1.add_argument("rttm",   help="входной .rttm файл")
    p1.add_argument("output", help="выходной .txt файл для Audacity")

    p2 = sub.add_parser("to_rttm", help="Audacity labels → RTTM")
    p2.add_argument("labels",   help="входной .txt файл из Audacity")
    p2.add_argument("output",   help="выходной .rttm файл")
    p2.add_argument("--file_id", default="audio", help="имя файла в RTTM (default: audio)")

    args = parser.parse_args()

    if args.cmd == "to_audacity":
        rttm_to_audacity(args.rttm, args.output)
    else:
        audacity_to_rttm(args.labels, args.output, args.file_id)
