import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
from experimental.rg_prefix_cache import LocalSuffixCache


@pytest.mark.parametrize('prefix',[0,3,256])
def test_external_length_does_not_allocate_or_append_external_gpu_cache(prefix):
    cache=LocalSuffixCache(prefix)
    assert cache.get_seq_length()==prefix
    assert cache.local_seq_length()==0
    assert cache.get_mask_sizes(2,0)==(prefix+2,0)
    q=torch.ones(1,2,2,8)
    k,v=cache.update(q,q,0)
    assert k.shape[-2]==2 and v.shape[-2]==2
    assert cache.get_seq_length()==prefix+2
    assert cache.local_seq_length()==2
    assert cache.get_mask_sizes(1,0)==(prefix+3,0)
    cache.update(q[:,:,:1],q[:,:,:1],0)
    assert cache.local_seq_length()==3


def test_bad_prefix_rejected():
    with pytest.raises(ValueError):
        LocalSuffixCache(-1)
