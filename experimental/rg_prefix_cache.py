"""GPU-local dynamic cache with positions accounting for an external prefix.

External KV ownership belongs to the R attention caller, not this GPU cache.
Only dense full attention is supported; no sliding, beam reordering or batching.
"""
from transformers import DynamicCache


class LocalSuffixCache(DynamicCache):
    def __init__(self, prefix_tokens):
        if not isinstance(prefix_tokens,int) or prefix_tokens<0:
            raise ValueError('nonnegative integer prefix length required')
        super().__init__()
        self.prefix_tokens=prefix_tokens

    def get_seq_length(self,layer_idx=0):
        return self.prefix_tokens+super().get_seq_length(layer_idx)

    def get_mask_sizes(self,query_length,layer_idx):
        local_length,offset=super().get_mask_sizes(query_length,layer_idx)
        if offset:
            raise ValueError('only dense non-sliding attention supported')
        return local_length+self.prefix_tokens,0

    def local_seq_length(self,layer_idx=0):
        return super().get_seq_length(layer_idx)
