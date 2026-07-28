import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.orchestrator.env_server.env_server import run_server
from prime_rl.orchestrator.envs import Env, Envs


def _factory_env() -> EnvConfig:
    return EnvConfig(
        name="apex",
        factory={
            "import_path": "harness.environment.load_environment",
            "kwargs": {"taskset": "harbor"},
        },
    )


def test_spawned_server_receives_factory_data(tmp_path):
    address_queue = MagicMock()
    address_queue.get.return_value = "tcp://127.0.0.1:12345"
    process = MagicMock()
    context = MagicMock()
    context.Queue.return_value = address_queue
    context.Process.return_value = process
    env = Env(_factory_env())

    with patch("prime_rl.orchestrator.envs.mp.get_context", return_value=context):
        address = asyncio.run(env._spawn(tmp_path, "INFO", False))

    assert address == "tcp://127.0.0.1:12345"
    process.start.assert_called_once_with()
    kwargs = context.Process.call_args.kwargs["kwargs"]
    assert kwargs["legacy"] is False
    assert kwargs["factory_path"] == "harness.environment.load_environment"
    assert kwargs["factory_kwargs"] == {"taskset": "harbor"}
    assert "config" not in kwargs


def test_standalone_server_receives_factory_data():
    config = EnvServerConfig(env=_factory_env())

    with patch("prime_rl.orchestrator.env_server.env_server.serve_env") as serve_env:
        run_server.__wrapped__(config)

    kwargs = serve_env.call_args.kwargs
    assert kwargs["legacy"] is False
    assert kwargs["factory_path"] == "harness.environment.load_environment"
    assert kwargs["factory_kwargs"] == {"taskset": "harbor"}
    assert "config" not in kwargs


def test_spawned_server_surfaces_child_startup_failure(tmp_path):
    address_queue = MagicMock()
    address_queue.get.side_effect = queue.Empty
    process = MagicMock()
    process.is_alive.return_value = False
    process.exitcode = 1
    context = MagicMock()
    context.Queue.return_value = address_queue
    context.Process.return_value = process
    env = Env(_factory_env())

    with (
        patch("prime_rl.orchestrator.envs.mp.get_context", return_value=context),
        patch("prime_rl.orchestrator.envs.ENV_SERVER_SPAWN_TIMEOUT", 0.2),
        pytest.raises(
            RuntimeError,
            match=r"Env server apex failed during startup \(exit code 1\)",
        ),
    ):
        asyncio.run(env._spawn(tmp_path, "INFO", False))

    process.join.assert_called()
    assert env._env_server_process is None


def test_environment_collection_unwinds_started_servers(tmp_path):
    process = MagicMock()
    process.is_alive.return_value = False
    started = MagicMock()
    started._env_server_process = process
    started.start = AsyncMock()
    failed = MagicMock()
    failed._env_server_process = None
    failed.start = AsyncMock(side_effect=RuntimeError("factory failed"))
    envs = Envs()
    envs._envs = {"started": started, "failed": failed}

    with pytest.raises(RuntimeError, match="factory failed"):
        asyncio.run(envs.start(tmp_path))

    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=25)
    assert started._env_server_process is None


def test_environment_collection_unwinds_on_cancellation(tmp_path):
    process = MagicMock()
    process.is_alive.return_value = False
    started = MagicMock()
    started._env_server_process = process
    started.start = AsyncMock()
    cancelled = MagicMock()
    cancelled._env_server_process = None
    cancelled.start = AsyncMock(side_effect=asyncio.CancelledError)
    envs = Envs()
    envs._envs = {"started": started, "cancelled": cancelled}

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(envs.start(tmp_path))

    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=25)
    assert started._env_server_process is None
