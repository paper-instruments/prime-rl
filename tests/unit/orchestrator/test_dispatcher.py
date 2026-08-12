import asyncio
import uuid
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

from verifiers.v1.clients.config import TrainClientConfig

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import GroupState, Policy
from prime_rl.utils.client import client_identity


async def _scheduled_clients(client_assignment: str):
    gate = asyncio.Event()
    seen_clients = []
    clients = [
        TrainClientConfig(base_url="http://worker:8000/v1", headers={"X-data-parallel-rank": str(rank)})
        for rank in range(2)
    ]
    pool = SimpleNamespace(
        model_name="model",
        select_train_client=AsyncMock(side_effect=clients),
    )

    async def run(**kwargs):
        seen_clients.append(kwargs["client"])
        await gate.wait()

    env = SimpleNamespace(
        sampler=SimpleNamespace(pool=pool, samples_from_live_policy=True),
        requires_group_scoring=False,
        run=run,
    )
    envs = SimpleNamespace(get=lambda _name: env)
    dispatcher = RolloutDispatcher(
        train_envs=envs,
        eval_envs=None,
        train_source=None,
        eval_source=None,
        policy_pool=pool,
        policy=Policy(version=1, model_name="model"),
        max_inflight_rollouts=2,
        tasks_per_minute=None,
        max_off_policy_steps=1,
        client_assignment=client_assignment,
    )
    group_id = uuid.uuid4()
    group = GroupState(
        kind="train",
        env_name="env",
        task_idx=0,
        rollouts_to_schedule=2,
        target_rollouts=2,
        policy_version_at_start=1,
    )
    dispatcher.groups[group_id] = group

    try:
        assert await dispatcher.schedule_group_rollout(group_id, group)
        assert await dispatcher.schedule_group_rollout(group_id, group)
        await asyncio.sleep(0)
        return dispatcher, group, pool, clients, seen_clients
    except BaseException:
        await dispatcher.cancel_inflight_rollouts()
        raise


def test_group_assignment_pins_all_rollouts_to_one_client():
    async def scenario():
        dispatcher, group, pool, clients, seen_clients = await _scheduled_clients("group")
        try:
            assert seen_clients == [clients[0], clients[0]]
            assert group.pinned_client == clients[0]
            pool.select_train_client.assert_awaited_once_with(Counter())
        finally:
            await dispatcher.cancel_inflight_rollouts()

    asyncio.run(scenario())


def test_trajectory_assignment_rebalances_each_rollout():
    async def scenario():
        dispatcher, group, pool, clients, seen_clients = await _scheduled_clients("trajectory")
        try:
            assert seen_clients == clients
            assert group.pinned_client is None
            assert pool.select_train_client.await_count == 2
            second_load = pool.select_train_client.await_args_list[1].args[0]
            assert second_load == Counter({client_identity(clients[0]): 1})
        finally:
            await dispatcher.cancel_inflight_rollouts()

    asyncio.run(scenario())
