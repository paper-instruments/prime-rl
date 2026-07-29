import pickle
from typing import Any, cast

import torch

from prime_rl.inference.vllm.worker.nccl import receive_integer, receive_state_dict


class _BroadcastQueue:
    def __init__(self, *values: torch.Tensor) -> None:
        self.device = torch.device("cpu")
        self._values = list(values)
        self.received_devices: list[torch.device] = []

    def broadcast(self, tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0
        self.received_devices.append(tensor.device)
        tensor.copy_(self._values.pop(0))


def test_nccl_metadata_allocates_on_communicator_device_under_meta_context():
    metadata = pickle.dumps({})
    integer_communicator = _BroadcastQueue(torch.tensor([7], dtype=torch.long))
    state_communicator = _BroadcastQueue(
        torch.tensor([len(metadata)], dtype=torch.long),
        torch.tensor(list(metadata), dtype=torch.uint8),
    )

    with torch.device("meta"):
        integer = receive_integer(cast(Any, integer_communicator))
        state = list(receive_state_dict(cast(Any, state_communicator)))

    assert integer == 7
    assert state == []
    assert integer_communicator.received_devices == [torch.device("cpu")]
    assert state_communicator.received_devices == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
