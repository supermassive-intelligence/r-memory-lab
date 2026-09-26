"""Lossless transport layout for a single coherent document-prefix KV cache."""
import torch


def pack_prefix(pairs, chunk_size=256):
    if not pairs or chunk_size < 1:
        raise ValueError('nonempty layer list and positive chunk size required')
    shape = pairs[0][0].shape
    if len(shape) != 4 or shape[0] != 1 or shape[2] < 1:
        raise ValueError('expected one nonempty B,H,T,D prefix')
    dtype = pairs[0][0].dtype
    for pair in pairs:
        if len(pair) != 2 or any(x.shape != shape or x.dtype != dtype or x.device.type != 'cpu' for x in pair):
            raise ValueError('all CPU K/V layers must share shape and dtype')
    # LMCache engine-driven layout: 2,L,T,H*D, preserving head ordering.
    packed = torch.stack([torch.stack([p[j][0].transpose(0,1) for p in pairs])
                          for j in range(2)]).flatten(-2).contiguous()
    chunks = []
    for offset in range(0,shape[2],chunk_size):
        chunk = torch.zeros(2,len(pairs),chunk_size,shape[1]*shape[3],dtype=dtype)
        count = min(chunk_size,shape[2]-offset)
        chunk[:,:,:count,:] = packed[:,:,offset:offset+count,:]
        chunks.append(chunk)
    metadata = dict(layers=len(pairs),heads=shape[1],tokens=shape[2],head_dim=shape[3],
                    dtype=str(dtype),chunk_size=chunk_size,chunks=len(chunks),layout='2,L,T,H*D')
    return chunks,metadata


def unpack_prefix(chunks, metadata):
    n,l,h,d,c = (metadata[k] for k in ('tokens','layers','heads','head_dim','chunk_size'))
    if min(n,l,h,d,c)<1 or len(chunks)!=(n+c-1)//c or metadata['chunks']!=len(chunks):
        raise ValueError('invalid valid length or chunk count')
    for chunk in chunks:
        if chunk.shape!=(2,l,c,h*d) or str(chunk.dtype)!=metadata['dtype'] or chunk.device.type!='cpu':
            raise ValueError('transport shape/dtype/device mismatch')
    packed=torch.cat(chunks,dim=2)[:,:,:n,:].reshape(2,l,n,h,d)
    return [(packed[0,i].transpose(0,1).unsqueeze(0).contiguous(),
             packed[1,i].transpose(0,1).unsqueeze(0).contiguous()) for i in range(l)]
