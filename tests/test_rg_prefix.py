import pytest
torch=pytest.importorskip('torch')
from experimental.rg_prefix import pack_prefix,unpack_prefix
from experimental.rg_numerics import bitwise_equal


@pytest.mark.parametrize('length',[1,255,256,257,513])
def test_transport_roundtrip_valid_length_and_layer_identity(length):
    gen=torch.Generator().manual_seed(18)
    pairs=[tuple(torch.randn(1,2,length,8,generator=gen).bfloat16() for _ in range(2)) for _ in range(3)]
    pairs[0][0][0,0,0,0]=-0.
    chunks,meta=pack_prefix(pairs)
    restored=unpack_prefix(chunks,meta)
    assert all(bitwise_equal(a,b) for p,q in zip(pairs,restored) for a,b in zip(p,q))
    if length%256:
        assert not chunks[-1][:,:,length%256:,:].count_nonzero()


def test_corrupt_metadata_or_shape_rejected():
    pairs=[(torch.zeros(1,2,3,8),torch.ones(1,2,3,8))]
    chunks,meta=pack_prefix(pairs)
    for altered in [dict(meta,tokens=0),dict(meta,tokens=257),dict(meta,heads=3),dict(meta,dtype='torch.bfloat16')]:
        with pytest.raises(ValueError):
            unpack_prefix(chunks,altered)
    with pytest.raises(ValueError):
        pack_prefix([])


def test_restart_port_guard_rejects_live_listener_but_accepts_time_wait():
    import socket
    from scripts.rg_lmcache_prefix import ensure_ports_available
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',0))
        port=server.getsockname()[1]
        server.listen()
        with pytest.raises(OSError):
            ensure_ports_available((port,))
        with socket.create_connection(('127.0.0.1',port)) as client:
            accepted,_=server.accept()
            accepted.close()
            assert client.recv(1)==b''
    ensure_ports_available((port,))


def test_l2_guard_counts_multichunk_delivery_and_rejects_duplicates():
    from scripts.rg_lmcache_prefix import validate_l2_reads
    log='(0 L1, 1 L2)\n(0 L1, 2 L2)\n'
    assert validate_l2_reads(log,[1,2])
    assert not validate_l2_reads(log,[1,1])
    assert not validate_l2_reads(log+'(0 L1, 2 L2)\n',[1,2])
    assert not validate_l2_reads('(1 L1, 0 L2)\n(0 L1, 2 L2)',[1,2])


def test_service_ports_preserve_default_and_isolate_second_instance():
    from scripts.rg_lmcache_prefix import service_ports
    assert service_ports()==(18580,18581)
    assert service_ports(19580)==(19580,19581)
    for invalid in (True,0,1023,65535,'19580'):
        with pytest.raises(ValueError):service_ports(invalid)
