"""Known BF16 midpoint fixtures; no change to model arithmetic."""
import torch
from experimental.rg_acceptance import require

def boundary_cases(codes):
    require(codes.dtype==torch.int16 and bool(((codes>=0)&(codes<0x7f7f)).all()),'finite adjacent positive BF16 codes required')
    low=codes.view(torch.bfloat16);high=(codes+1).view(torch.bfloat16)
    midpoint=(low.double()+high.double())/2
    before=torch.nextafter(midpoint,torch.full_like(midpoint,-torch.inf))
    after=torch.nextafter(midpoint,torch.full_like(midpoint,torch.inf))
    tie=torch.where((codes&1)==0,low,high)
    inputs=torch.stack((before,midpoint,after))
    expected=torch.stack((low,tie,high))
    return torch.cat((inputs,-inputs)),torch.cat((expected,-expected))
