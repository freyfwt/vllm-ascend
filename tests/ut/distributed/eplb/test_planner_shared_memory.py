import numpy as np

from vllm_ascend.distributed.eplb.planner_shared_memory import (
    SharedSnapshotBuffer,
    SharedSnapshotShape,
)


def test_shared_snapshot_round_trip_and_slot_sequence():
    owner = SharedSnapshotBuffer.create(SharedSnapshotShape(4, 2, 3, 2, 2))
    attached = SharedSnapshotBuffer.attach(owner.spec)
    values = (
        np.arange(12, dtype=np.int64).reshape(2, 2, 3),
        np.array([2, 3], dtype=np.int64),
        np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.int32),
        np.array([4, 5], dtype=np.int64),
        np.array([1.1, np.nan]),
    )
    try:
        slot, sequence, digest = owner.write(*values)
        assert (slot, sequence) == (0, 1)
        actual = attached.read(slot, 2, digest)
        for expected, result in zip(values, actual):
            np.testing.assert_equal(result, expected)
        assert owner.write(*values)[:2] == (1, 2)
    finally:
        attached.close()
        owner.close()


def test_shared_snapshot_detects_corruption():
    owner = SharedSnapshotBuffer.create(SharedSnapshotShape(1, 1, 1, 1, 1))
    try:
        slot, _, digest = owner.write(
            np.ones((1, 1, 1), dtype=np.int64),
            np.ones(1, dtype=np.int64),
            np.zeros((1, 1, 1), dtype=np.int32),
            np.zeros(1, dtype=np.int64),
            np.full(1, np.nan),
        )
        owner._arrays(slot)[0][0, 0, 0] = 2
        try:
            owner.read(slot, 1, digest)
            raise AssertionError("corruption was not detected")
        except RuntimeError:
            pass
    finally:
        owner.close()
