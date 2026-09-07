import pytest

from vllm_ascend.distributed.eplb.topology import build_topology


def test_topology_uses_explicit_global_rank_bijection_and_host_identity():
    topology = build_topology([4, 6, 8, 10], [8, 10, 4, 6], [b"a", b"a", b"b", b"b"])
    assert topology.node_by_ep_rank == (0, 0, 1, 1)
    assert topology.eplb_rank_by_ep_rank == (2, 3, 0, 1)


def test_topology_rejects_mismatched_process_groups():
    with pytest.raises(ValueError, match="same global ranks"):
        build_topology([0, 1], [0, 2], [b"a", b"b"])
