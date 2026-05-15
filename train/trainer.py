import os
import sys
import time
import torch
import torch.nn as nn

from models.ecapa_tdnn import ECAPA_TDNN
from models.loss import AAMsoftmax
from utils.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super().__init__()
        encoder = ECAPA_TDNN(C=C).cuda()
        if torch.cuda.device_count() > 1:
            print(f"DataParallel: используем {torch.cuda.device_count()} GPU")
            encoder = nn.DataParallel(encoder)
        self.speaker_encoder = encoder
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=test_step, gamma=lr_decay
        )
        n_params = sum(p.numel() for p in self.speaker_encoder.parameters()) / 1024 / 1024
        print(time.strftime("%m-%d %H:%M:%S") + f" Параметров модели: {n_params:.2f}M")

    def train_network(self, epoch, loader, save_path=None, save_every=500):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]["lr"]
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(
                time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Batch: %d, Loss: %.5f, ACC: %2.2f%%\r"
                % (epoch, lr, num, loss / num, top1 / index * len(labels))
            )
            sys.stderr.flush()
            if save_path and num % save_every == 0:
                ckpt = os.path.join(save_path, "model_%04d.model" % epoch)
                self.save_parameters(ckpt)
                sys.stdout.write("\n[ckpt] batch %d -> %s\n" % (num, ckpt))
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
            if name not in self_state:
                name = origname.replace("speaker_encoder.", "speaker_encoder.module.")
            if name not in self_state:
                print(f"{origname} не найден в модели.")
                continue
            if self_state[name].size() != param.size():
                print(f"Размер не совпадает: {origname}, "
                      f"модель: {self_state[name].size()}, загружено: {param.size()}")
                continue
            self_state[name].copy_(param)
