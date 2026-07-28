import asyncio
import uuid
from collections import defaultdict
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Rollout


def test_atomic_group_rejects_every_member_when_one_trace_errors():
    algorithm = _Algorithm()
    sink = _sink(atomic_group=True, algorithm=algorithm)
    group_id = uuid.uuid4()
    successful = _rollout(group_id, rollout_group_id="apex-group")
    failed = _rollout(
        group_id,
        rollout_group_id="apex-group",
        error=vf.Error(type="HarnessError", message="sandbox failed"),
    )
    sink.pending_groups[group_id] = [successful, failed]

    asyncio.run(sink.process_group(group_id))

    assert sink.pending_batch == []
    assert list(sink.pending_rollouts) == [successful, failed]
    assert algorithm.finalized_groups == []
    assert successful.info["rollout_group_id"] == "apex-group"
    assert failed.info["rollout_group_id"] == "apex-group"
    assert successful.rewards == {}
    assert failed.rewards == {}
    assert failed.errors == [vf.Error(type="HarnessError", message="sandbox failed")]


def test_default_non_atomic_group_keeps_successful_siblings():
    algorithm = _Algorithm()
    sink = _sink(atomic_group=False, algorithm=algorithm)
    group_id = uuid.uuid4()
    successful = _rollout(group_id, rollout_group_id="environment-group")
    failed = _rollout(
        group_id,
        rollout_group_id="environment-group",
        error=vf.Error(type="ProviderError", message="request failed"),
    )
    sink.pending_groups[group_id] = [successful, failed]

    asyncio.run(sink.process_group(group_id))

    assert sink.pending_batch == [successful]
    assert algorithm.finalized_groups == [[successful]]
    assert successful.info["rollout_group_id"] == "environment-group"


def test_group_scoring_partial_remains_atomic_without_new_option():
    algorithm = _Algorithm()
    sink = _sink(
        atomic_group=False,
        algorithm=algorithm,
        requires_group_scoring=True,
    )
    group_id = uuid.uuid4()
    successful = _rollout(group_id)
    failed = _rollout(
        group_id,
        error=vf.Error(type="ScoringError", message="group incomplete"),
    )
    sink.pending_groups[group_id] = [successful, failed]

    asyncio.run(sink.process_group(group_id))

    assert sink.pending_batch == []
    assert algorithm.finalized_groups == []


class _Algorithm:
    def __init__(self):
        self.finalized_groups = []

    async def finalize_group(self, group):
        self.finalized_groups.append(group)


class _TrainEnvs:
    def __init__(self, env):
        self.env = env

    def get(self, name):
        assert name == "env"
        return self.env


def _sink(
    *,
    atomic_group,
    algorithm,
    requires_group_scoring=False,
):
    sink = TrainSink.__new__(TrainSink)
    sink.train_envs = _TrainEnvs(
        SimpleNamespace(
            config=SimpleNamespace(group_size=2, atomic_group=atomic_group),
            requires_group_scoring=requires_group_scoring,
            algorithm=algorithm,
            sampling_args={"temperature": 1.0},
        )
    )
    sink.pending_groups = defaultdict(list)
    sink.pending_rollouts = TrainRollouts()
    sink.pending_batch = []
    sink.pending_tokens = 0
    sink.token_batch_size = None
    sink.pre_filters = []
    sink.pre_filter_seen = 0
    sink.pre_filter_dropped = 0
    sink.pre_filter_dropped_by_name = {}
    return sink


def _rollout(
    group_id,
    *,
    rollout_group_id="",
    error=None,
):
    return Rollout(
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
        ),
        env_name="env",
        group_id=group_id,
        info={"rollout_group_id": rollout_group_id},
        errors=[error] if error is not None else [],
    )
