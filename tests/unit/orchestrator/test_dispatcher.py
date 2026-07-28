import asyncio
import uuid
from collections import deque
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError

from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import GroupState, Policy, Rollout


def test_atomic_group_is_default_off_and_roundtrips():
    default = EnvConfig(
        name="default",
        factory={"import_path": "package.load_environment"},
    )
    atomic = EnvConfig(
        name="atomic",
        factory={"import_path": "package.load_environment"},
        atomic_group=True,
    )

    assert default.atomic_group is False
    assert atomic.atomic_group is True
    assert EnvConfig.model_validate(atomic.model_dump()) == atomic


@pytest.mark.parametrize("section", ["train", "eval"])
def test_atomic_group_must_fit_shared_rollout_capacity(section):
    config = {
        "renderer": {"name": "qwen3"},
        "max_inflight_rollouts": 2,
        section: {
            "env": [
                {
                    "id": "environment",
                    "atomic_group": True,
                    "group_size": 3,
                }
            ]
        },
    }

    with pytest.raises(ValidationError, match="group_size=3.*max_inflight_rollouts=2"):
        OrchestratorConfig.model_validate(config)


@pytest.mark.parametrize(
    ("requires_group_scoring", "atomic_group", "expected_call", "expected_permits"),
    [
        (False, False, "rollout", 1),
        (True, False, "group", 3),
        (False, True, "group", 3),
    ],
)
def test_dispatch_path_and_permits_share_the_group_predicate(
    requires_group_scoring: bool,
    atomic_group: bool,
    expected_call: str,
    expected_permits: int,
):
    calls = []
    pool = _Pool()
    env = _Env(
        pool,
        calls,
        requires_group_scoring=requires_group_scoring,
        atomic_group=atomic_group,
    )
    envs = _Envs(env)
    dispatcher = RolloutDispatcher(
        train_envs=envs,
        eval_envs=None,
        train_source=SimpleNamespace(),
        eval_source=None,
        policy_pool=pool,
        policy=Policy(version=7, model_name="policy"),
        max_inflight_rollouts=8,
        tasks_per_minute=None,
        max_off_policy_steps=1,
    )
    group_id = uuid.uuid4()
    group = GroupState(
        kind="train",
        env_name=env.name,
        task_idx=11,
        rollouts_to_schedule=3,
        target_rollouts=3,
        policy_version_at_start=7,
    )
    dispatcher.groups[group_id] = group

    async def schedule():
        assert await dispatcher.schedule_group_rollout(group_id, group)
        task = next(iter(dispatcher.inflight))
        await task

    asyncio.run(schedule())

    assert calls == [(expected_call, 3 if expected_call == "group" else None)]
    assert dispatcher.inflight_permits == expected_permits
    assert next(iter(dispatcher.inflight.values())).rollout_count == expected_permits
    assert group.rollouts_to_schedule == (0 if expected_call == "group" else 2)


def test_atomic_group_completion_preserves_traces_and_releases_permits():
    group_id = uuid.uuid4()
    successful = _rollout(rollout_group_id="apex-group")
    failed = _rollout(
        rollout_group_id="apex-group",
        error=vf.Error(type="HarnessError", message="sandbox failed"),
    )
    pool = _Pool()
    env = _Env(
        pool,
        [],
        atomic_group=True,
        results=[successful, failed, _rollout(rollout_group_id="apex-group")],
    )
    dispatcher = RolloutDispatcher(
        train_envs=_Envs(env),
        eval_envs=None,
        train_source=SimpleNamespace(),
        eval_source=None,
        policy_pool=pool,
        policy=Policy(version=7, model_name="policy"),
        max_inflight_rollouts=3,
        tasks_per_minute=None,
        max_off_policy_steps=1,
    )
    group = GroupState(
        kind="train",
        env_name=env.name,
        task_idx=11,
        rollouts_to_schedule=3,
        target_rollouts=3,
        policy_version_at_start=7,
    )
    dispatcher.groups[group_id] = group

    async def schedule_and_complete():
        assert await dispatcher.schedule_group_rollout(group_id, group)
        task = next(iter(dispatcher.inflight))
        await task
        await dispatcher.handle_completed_rollout(task)

    asyncio.run(schedule_and_complete())
    emitted = [dispatcher.out_q.get_nowait() for _ in range(3)]

    assert dispatcher.inflight_permits == 0
    assert dispatcher.inflight == {}
    assert group_id not in dispatcher.groups
    assert [rollout.group_id for rollout in emitted] == [group_id] * 3
    assert [rollout.info["rollout_group_id"] for rollout in emitted] == ["apex-group"] * 3
    assert emitted[1].errors == [vf.Error(type="HarnessError", message="sandbox failed")]


@pytest.mark.parametrize(
    ("requires_group_scoring", "atomic_group", "available", "is_scheduled"),
    [
        (False, False, 1, True),
        (True, False, 2, False),
        (False, True, 2, False),
        (False, True, 3, True),
    ],
)
def test_train_source_reserves_complete_group_capacity(
    requires_group_scoring: bool,
    atomic_group: bool,
    available: int,
    is_scheduled: bool,
):
    env = _SourceEnv(
        requires_group_scoring=requires_group_scoring,
        atomic_group=atomic_group,
    )
    source = TrainSource(_Envs(env), seed=0)

    result = source.next_example(available)

    assert (result is not None) is is_scheduled
    assert source.cursors[env.name] == int(is_scheduled)


@pytest.mark.parametrize(
    ("requires_group_scoring", "atomic_group", "available", "is_scheduled"),
    [
        (False, False, 1, True),
        (True, False, 2, False),
        (False, True, 2, False),
        (False, True, 3, True),
    ],
)
def test_eval_source_reserves_complete_group_capacity(
    requires_group_scoring: bool,
    atomic_group: bool,
    available: int,
    is_scheduled: bool,
):
    env = _SourceEnv(
        requires_group_scoring=requires_group_scoring,
        atomic_group=atomic_group,
    )
    source = EvalSource.__new__(EvalSource)
    source.eval_envs = _Envs(env)
    source.queue = deque([{"env_name": env.name, "task_idx": 4}])

    result = source.next_example(available)

    assert (result is not None) is is_scheduled
    assert len(source.queue) == (0 if is_scheduled else 1)


class _Pool:
    model_name = "policy"

    async def select_train_client(self, load):
        return SimpleNamespace()


class _Env:
    name = "env"
    num_tasks = 1

    def __init__(
        self,
        pool,
        calls,
        *,
        requires_group_scoring=False,
        atomic_group=False,
        results=None,
    ):
        self.requires_group_scoring = requires_group_scoring
        self.config = SimpleNamespace(
            atomic_group=atomic_group,
            group_size=3,
            ratio=1.0,
        )
        self.sampler = SimpleNamespace(
            samples_from_live_policy=True,
            pool=pool,
        )
        self.calls = calls
        self.results = results

    async def run_group(
        self,
        *,
        client,
        task_idx,
        model_name,
        group_size,
        cache_salt,
    ):
        self.calls.append(("group", group_size))
        return self.results or []

    async def run_rollout(
        self,
        *,
        client,
        task_idx,
        model_name,
        cache_salt,
    ):
        self.calls.append(("rollout", None))
        return SimpleNamespace()


class _SourceEnv(_Env):
    def __init__(self, *, requires_group_scoring, atomic_group):
        super().__init__(
            _Pool(),
            [],
            requires_group_scoring=requires_group_scoring,
            atomic_group=atomic_group,
        )


class _Envs:
    def __init__(self, env):
        self.env = env

    def __iter__(self):
        yield self.env

    def get(self, name):
        assert name == self.env.name
        return self.env


def _rollout(*, rollout_group_id, error=None):
    nodes = []
    if error is None:
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content="done"),
                sampled=True,
                token_ids=[1],
                mask=[True],
                logprobs=[-0.1],
            )
        )
    return Rollout(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=11, prompt=None)),
        nodes=nodes,
        info={"rollout_group_id": rollout_group_id},
        errors=[error] if error is not None else [],
    )
