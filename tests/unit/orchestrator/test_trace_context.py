import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.envs import Env
from prime_rl.orchestrator.types import GroupState, Policy


def test_dispatcher_sends_group_and_policy_version_to_env_server():
    async def run():
        recorded_request = None
        client_config = object()

        class EnvClient:
            async def run(self, **request):
                nonlocal recorded_request
                recorded_request = request
                return SimpleNamespace(traces=[], last_error=None)

        class InferencePool:
            model_name = "policy"

            async def select_train_client(self, _load):
                return client_config

        pool = InferencePool()
        env = Env(SimpleNamespace(resolved_name="env"), address="unused")
        env._env_client = EnvClient()
        env.sampler = SimpleNamespace(pool=pool, samples_from_live_policy=True)
        envs = SimpleNamespace(get=lambda _name: env)
        dispatcher = RolloutDispatcher(
            train_envs=envs,
            eval_envs=None,
            train_source=object(),
            eval_source=None,
            policy_pool=pool,
            policy=Policy(version=7, model_name="policy"),
            max_inflight_episodes=1,
            tasks_per_minute=None,
            max_off_policy_steps=1,
        )
        group_id = uuid.uuid4()
        group = GroupState(
            kind="train",
            env_name="env",
            task_idx=3,
            rollouts_to_schedule=1,
            target_rollouts=1,
            policy_version_at_start=7,
        )
        dispatcher.groups[group_id] = group

        assert await dispatcher.schedule_group_rollout(group_id, group)
        task = next(iter(dispatcher.inflight))
        with pytest.raises(RuntimeError, match="episode failed before any trace"):
            await task

        assert recorded_request is not None
        assert recorded_request["trace_info"] == {
            "rollout_group_id": str(group_id),
            "sampled_policy_version": 7,
        }

    asyncio.run(run())
