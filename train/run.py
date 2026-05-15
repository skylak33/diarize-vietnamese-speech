import argparse
import glob
import os
import sys
import time
import warnings

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import init_args
from train.loader_hf import TrainLoader
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

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")
args = parser.parse_args()
args = init_args(args)

trainloader = TrainLoader(**vars(args))
trainLoader = torch.utils.data.DataLoader(
    trainloader,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

modelfiles = sorted(glob.glob("%s/model_0*.model" % args.model_save_path))

if args.eval:
    s = ECAPAModel(**vars(args))
    s.load_parameters(args.initial_model)
    print("Модель загружена: %s" % args.initial_model)
    quit()

if args.initial_model:
    s = ECAPAModel(**vars(args))
    s.load_parameters(args.initial_model)
    print("Продолжаем с: %s" % args.initial_model)
    epoch = 1
elif modelfiles:
    s = ECAPAModel(**vars(args))
    s.load_parameters(modelfiles[-1])
    print("Продолжаем с: %s" % modelfiles[-1])
    epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
else:
    s = ECAPAModel(**vars(args))
    epoch = 1

score_file = open(args.score_save_path, "a+")

while True:
    loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader,
                                    save_path=args.model_save_path)
    if epoch % args.test_step == 0:
        s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              "%d epoch, ACC %2.2f%%, LR %f, LOSS %f" % (epoch, acc, lr, loss))
        score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%\n" % (epoch, lr, loss, acc))
        score_file.flush()

    if epoch >= args.max_epoch:
        break
    epoch += 1
