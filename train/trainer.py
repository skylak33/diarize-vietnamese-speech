import os
import sys
import time
import torch
import torch.nn as nn

from models.ecapa_tdnn import ECAPA_TDNN
from models.loss import AAMsoftmax
from utils.device import resolve_device


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super().__init__()
        self.device = resolve_device(kwargs.get("device", "auto"))
        self.specaug = kwargs.get("augmentation") != "none"
        encoder = ECAPA_TDNN(C=C).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"DataParallel: используем {torch.cuda.device_count()} GPU")
            encoder = nn.DataParallel(encoder)
        self.speaker_encoder = encoder
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=test_step, gamma=lr_decay
        )
        n_params = sum(p.numel() for p in self.speaker_encoder.parameters()) / 1024 / 1024
        print(time.strftime("%m-%d %H:%M:%S") + f" Параметров модели: {n_params:.2f}M")

    def train_network(self, epoch, loader, save_path=None, save_every=500):
        self.train()
        # Epoch-indexed StepLR schedule also works for weight-only continuation.
        for group, base_lr in zip(self.optim.param_groups, self.scheduler.base_lrs):
            group['lr'] = base_lr * self.scheduler.gamma ** ((epoch - 1) // self.scheduler.step_size)
        index, top1, loss, num = 0, 0, 0, 0
        lr = self.optim.param_groups[0]["lr"]
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = labels.to(self.device, dtype=torch.long)
            speaker_embedding = self.speaker_encoder(data.to(self.device), aug=self.specaug)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += float(prec.item()) * len(labels)
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(
                time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Batch: %d, Loss: %.5f, ACC: %2.2f%%\r"
                % (epoch, lr, num, loss / num, top1 / index)
            )
            sys.stderr.flush()
            if save_path and num % save_every == 0:
                ckpt = os.path.join(save_path, "model_%04d.model" % epoch)
                self.save_parameters(ckpt)
                sys.stdout.write("\n[ckpt] batch %d -> %s\n" % (num, ckpt))
        sys.stdout.write("\n")
        if not num:
            raise ValueError("Training loader produced no full batches; check data/batch_size")
        return float(loss / num), lr, top1 / index

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)
        loader = getattr(self, "training_loader", None)
        if loader is not None and hasattr(loader, "speaker_map"):
            import json
            from pathlib import Path
            Path(str(path) + ".speakers.json").write_text(json.dumps(loader.speaker_map, sort_keys=True), encoding="utf-8")

    def load_parameters(self, path, encoder_only=False):
        from utils.model import load_model
        loaded_encoder = load_model(path, 'cpu')
        target = self.speaker_encoder.module if isinstance(self.speaker_encoder, nn.DataParallel) else self.speaker_encoder
        target.load_state_dict(loaded_encoder.state_dict(), strict=True)
        if encoder_only:
            return
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        loss_state = {k.removeprefix('module.').removeprefix('speaker_loss.'): v
                      for k, v in state.items() if k.removeprefix('module.').startswith('speaker_loss.')}
        if loss_state:
            self.speaker_loss.load_state_dict(loss_state, strict=True)
