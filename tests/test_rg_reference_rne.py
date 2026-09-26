import torch
import pytest
from experimental.rg_reference_rne import direct_bf16
from scripts.rg_reference_conversion import boundary_cases


def test_all_finite_boundary_neighbors_and_ties():
    x,y=boundary_cases(torch.arange(0,0x7f7f,dtype=torch.int16))
    assert torch.equal(direct_bf16(x).view(torch.int16),y.view(torch.int16))


def test_all_finite_endpoints_and_signed_zero():
    codes=torch.arange(0,0x7f80,dtype=torch.int16)
    x=codes.view(torch.bfloat16)
    for values in (x,-x):
        assert torch.equal(direct_bf16(values.double()).view(torch.int16),values.view(torch.int16))


def test_overflow_and_special_values():
    threshold=float.fromhex('0x1.ffp127')
    t=torch.tensor(threshold,dtype=torch.float64)
    below=torch.nextafter(t,torch.tensor(0.,dtype=torch.float64))
    x=torch.stack((below,t,-below,-t,torch.tensor(float('inf')),torch.tensor(float('-inf')))).double()
    assert direct_bf16(x).view(torch.int16).tolist()==[0x7f7f,0x7f80,-129,-128,0x7f80,-128]
    assert torch.isnan(direct_bf16(torch.tensor(float('nan'),dtype=torch.float64)))


def test_captured_midpoint_value():
    x=torch.tensor(1.8554687471433955,dtype=torch.float64)
    assert direct_bf16(x).item()==1.8515625


def test_dtype_required():
    with pytest.raises(ValueError):direct_bf16(torch.tensor([1.]))
