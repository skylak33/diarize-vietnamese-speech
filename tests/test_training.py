import numpy as np
import torch
import soundfile as sf
import pytest
from train.loader_vlsp import VLSPLoader
from train.loader_hf import TrainLoader
from train.trainer import ECAPAModel
from utils.model import load_model
from utils.seed import set_seed


def test_local_loader_resamples_and_augmentation_is_used(tmp_path):
    sf.write(tmp_path/'a.wav', np.ones((8000,2))*.1,8000)
    metadata=tmp_path/'meta.txt'; metadata.write_text('A\ta.wav\n')
    loader=VLSPLoader(20,n_class=1,metadata_path=metadata,audio_base=tmp_path,augmentation='none')
    audio,label=next(iter(loader))
    assert label==0 and audio.shape==(3440,)
    assert torch.allclose(audio,torch.full_like(audio,.1),atol=.001)
    loader.augmenter=lambda x:x*0
    assert next(iter(loader))[0].abs().max()==0


def test_dolly_label_zero_and_capacity():
    loader=TrainLoader(20,n_class=1,augmentation='none')
    assert loader._get_label('0')==0
    with pytest.raises(ValueError,match='n_class'):
        loader._get_label('1')


def test_cpu_training_and_checkpoint_roundtrip(tmp_path):
    set_seed(42)
    torch.set_num_threads(1)
    model=ECAPAModel(lr=.001,lr_decay=.97,C=8,n_class=2,m=.2,s=30,test_step=1,device='cpu',augmentation='none')
    loss,lr,acc=model.train_network(1,[(torch.randn(2,3440),torch.tensor([0,1]))])
    assert np.isfinite(loss) and 0<=acc<=100
    path=tmp_path/'model.model'; model.save_parameters(path)
    encoder=load_model(path,'cpu')
    assert encoder.conv1.weight.shape[0]==8
    # DataParallel prefix compatibility without requiring CUDA.
    state={'speaker_encoder.module.'+k:v for k,v in encoder.state_dict().items()}
    torch.save(state,path)
    assert load_model(path,'cpu').training is False


def test_empty_training_loader():
    model=ECAPAModel(lr=.001,lr_decay=.97,C=8,n_class=2,m=.2,s=30,test_step=1,device='cpu')
    with pytest.raises(ValueError,match='no full batches'):
        model.train_network(1,[])
