import random
import numpy as np
import pytest
import torch
from utils.seed import set_seed
from diarize.cluster import cluster_embeddings, merge_short_segments
from diarize.pipeline import merge_consecutive_segments, partition_overlapping_windows, extract_embeddings_for_segments


def test_seed_reproducibility():
    def sample():
        return random.random(), np.random.rand(), torch.rand(1).item()
    set_seed(17); a=sample()
    set_seed(17); assert sample() == a


def test_merge_consecutive():
    segs, labels=merge_consecutive_segments([(0,2),(1,1.5),(2,3),(4,5)],np.array([0,0,0,0]),0)
    assert segs == [(0,3),(4,5)]
    assert labels.tolist() == [0,0]
    assert merge_consecutive_segments([],np.array([]))[0] == []


def test_partition_preserves_speech_without_artificial_overlap():
    result=partition_overlapping_windows([(0,1.5),(.75,2.25),(1.5,3),(4,5)])
    assert sum(e-s for s,e in result) == 4
    assert all(a[1] <= b[0] for a,b in zip(result,result[1:]))
    assert merge_short_segments([(0,1),(2,2.1)],np.array([0,1]))[0] == [(0,1),(2,2.1)]


@pytest.mark.parametrize('method',['ahc','spectral'])
def test_deterministic_clustering(method):
    rng=np.random.default_rng(42)
    emb=np.vstack([rng.normal([1.,0.,0.],0.01,(8,3)),rng.normal([0.,1.,0.],0.01,(8,3))])
    a=cluster_embeddings(emb,method,n_speakers=2,seed=7)
    b=cluster_embeddings(emb,method,n_speakers=2,seed=7)
    assert np.array_equal(a,b)
    assert len(set(a[:8])) == len(set(a[8:])) == 1 and a[0] != a[8]
    assert cluster_embeddings(np.ones((2,3)),method).tolist() == [0,0]


def test_bad_window():
    with pytest.raises(ValueError):
        extract_embeddings_for_segments(None,np.zeros(16000),[(0,1)],step_sec=0)
