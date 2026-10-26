import numpy as np
import soundfile as sf
from train.augmentation import Augmenter
from utils.seed import set_seed


def test_waveform_augmentation(tmp_path):
    rng=np.random.default_rng(0)
    for kind in ('noise','music','speech'):
        folder=tmp_path/'musan'/kind/'source'
        folder.mkdir(parents=True)
        sf.write(folder/'a.wav',rng.normal(0,.1,16000),16000)
    rir=tmp_path/'rir'; rir.mkdir()
    sf.write(rir/'room.wav',np.array([1.,0.,.5]),16000)
    aug=Augmenter('full',tmp_path/'musan',rir)
    audio=np.sin(np.arange(8000)*.1).astype(np.float32)
    for kind in ('noise','music','speech'):
        noisy=aug.add_noise(audio,kind)
        assert noisy.shape == audio.shape and not np.array_equal(noisy,audio)
        assert np.isfinite(noisy).all()
    assert not np.array_equal(aug.add_rir(audio),audio)
    assert np.array_equal(Augmenter('none')(audio),audio)
    set_seed(42); a=aug(audio)
    set_seed(42); assert np.array_equal(aug(audio),a)
