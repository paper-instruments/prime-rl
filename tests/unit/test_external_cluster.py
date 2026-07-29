import os
import signal
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest
from pydantic import ValidationError

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.external_cluster import parse_args
from prime_rl.external_cluster import (
    ClusterControlClient,
    ClusterCoordinator,
    ComponentPlan,
    ExternalClusterError,
    LocalProcessSupervisor,
    NodeRole,
    build_external_node_plan,
    run_external_cluster,
    run_external_node,
    write_external_subconfigs,
)

_LAUNCH_IDENTITY = {
    "run_id": "test-run",
    "world_size": 2,
    "address_fingerprint": "addresses",
    "config_fingerprint": "config",
}


def external_config(tmp_path: Path, **overrides) -> RLConfig:
    data = {
        "output_dir": tmp_path / "output",
        "trainer": {"rollout_transport": {"type": "zmq", "port": 5555, "hwm": 10}},
        "orchestrator": {"rollout_transport": {"type": "zmq", "port": 5555, "hwm": 10}},
        "inference": {
            "parallel": {"tp": 1},
            "deployment": {"type": "single_node", "gpus_per_node": 1, "backend_port": 8100},
        },
        "deployment": {
            "type": "multi_node",
            "gpus_per_node": 1,
            "num_train_nodes": 2,
            "num_infer_nodes": 1,
        },
        "weight_broadcast": {"type": "nccl", "port": 29501},
        "external_cluster": {
            "trainer_rdzv_port": 29500,
            "consensus_port": 29503,
            "startup_timeout": 2,
            "shutdown_timeout": 2,
            "termination_grace_period": 0.05,
        },
    }
    data.update(overrides)
    return RLConfig.model_validate(data)


def test_external_cluster_role_mapping_and_process_contract(tmp_path):
    config = external_config(tmp_path)
    addresses = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    inference = build_external_node_plan(
        config,
        rank=0,
        addresses=addresses,
        run_id="run-123",
        local_state_dir=tmp_path / "rank-0",
        environ={"PATH": "/bin"},
    )
    trainer_zero = build_external_node_plan(
        config,
        rank=1,
        addresses=addresses,
        run_id="run-123",
        local_state_dir=tmp_path / "rank-1",
        environ={"PATH": "/bin"},
    )
    trainer_one = build_external_node_plan(
        config,
        rank=2,
        addresses=addresses,
        run_id="run-123",
        local_state_dir=tmp_path / "rank-2",
        environ={"PATH": "/bin"},
    )

    assert inference.role is NodeRole.INFERENCE
    assert inference.trainer_node_rank is None
    assert [component.name for component in inference.components] == ["inference"]
    assert trainer_zero.role is NodeRole.TRAINER
    assert trainer_zero.trainer_node_rank == 0
    assert [component.name for component in trainer_zero.components] == ["orchestrator", "trainer"]
    assert trainer_one.trainer_node_rank == 1
    assert [component.name for component in trainer_one.components] == ["trainer"]
    assert inference.address_fingerprint == trainer_zero.address_fingerprint == trainer_one.address_fingerprint
    assert inference.config_fingerprint == trainer_zero.config_fingerprint == trainer_one.config_fingerprint

    trainer_command = trainer_one.components[0]
    assert "--nnodes=2" in trainer_command.argv
    assert "--nproc-per-node=1" in trainer_command.argv
    assert "--node-rank=1" in trainer_command.argv
    assert "--rdzv-endpoint=10.0.0.2:29500" in trainer_command.argv
    assert "--rdzv-id=run-123" in trainer_command.argv
    assert trainer_command.env["CUDA_VISIBLE_DEVICES"] == "0"

    resolved = trainer_zero.resolved_config
    assert resolved.trainer.rollout_transport.host == "10.0.0.2"
    assert resolved.orchestrator.rollout_transport.host == "10.0.0.2"
    assert resolved.orchestrator.model.client.base_url == ["http://10.0.0.1:8100/v1"]
    assert resolved.orchestrator.model.client.admin_base_url == ["http://10.0.0.1:8100/v1"]
    assert resolved.orchestrator.weight_broadcast.host == "10.0.0.2"


def test_external_cluster_preserves_provider_gpu_visibility(tmp_path):
    config = external_config(
        tmp_path,
        deployment={
            "type": "multi_node",
            "gpus_per_node": 2,
            "num_train_nodes": 1,
            "num_infer_nodes": 1,
        },
        inference={
            "parallel": {"tp": 2},
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 2,
                "backend_port": 8100,
            },
        },
    )
    plan = build_external_node_plan(
        config,
        rank=1,
        addresses=["10.0.0.1", "10.0.0.2"],
        run_id="gpu-visibility",
        local_state_dir=tmp_path / "rank-1",
        environ={"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"},
    )

    assert plan.components[0].env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"


def test_external_cluster_rejects_incomplete_provider_gpu_visibility(tmp_path):
    config = external_config(tmp_path)

    with pytest.raises(ValueError, match="exactly 1 devices"):
        build_external_node_plan(
            config,
            rank=1,
            addresses=["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            run_id="gpu-visibility",
            local_state_dir=tmp_path / "rank-1",
            environ={"CUDA_VISIBLE_DEVICES": ""},
        )


@pytest.mark.parametrize("address", ["fd00::10", "trainer.internal", "10.0.0.999"])
def test_external_cluster_rejects_non_ipv4_addresses(tmp_path, address):
    config = external_config(tmp_path)
    with pytest.raises(ValueError, match="IPv4"):
        build_external_node_plan(
            config,
            rank=1,
            addresses=["10.0.0.1", address, "10.0.0.3"],
            run_id="invalid-address",
            local_state_dir=tmp_path / "rank-1",
            environ={},
        )


def test_external_cluster_writes_rank_local_and_one_writer_evidence(tmp_path):
    config = external_config(tmp_path)
    inference = build_external_node_plan(
        config,
        rank=0,
        addresses=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
        run_id="local-config",
        local_state_dir=tmp_path / "rank-0",
        environ={},
    )
    trainer_master = build_external_node_plan(
        config,
        rank=1,
        addresses=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
        run_id="local-config",
        local_state_dir=tmp_path / "rank-1",
        environ={},
    )

    write_external_subconfigs(inference)
    write_external_subconfigs(trainer_master)

    assert (tmp_path / "rank-0" / "configs" / "inference.toml").is_file()
    assert (tmp_path / "rank-0" / "configs" / "trainer.toml").is_file()
    assert (tmp_path / "rank-0" / "configs" / "orchestrator.toml").is_file()
    assert (config.output_dir / "configs" / "inference.toml").is_file()
    assert (config.output_dir / "configs" / "trainer.toml").is_file()
    assert (config.output_dir / "configs" / "orchestrator.toml").is_file()


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "."])
def test_external_cluster_rejects_unsafe_run_id(tmp_path, run_id):
    config = external_config(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        build_external_node_plan(
            config,
            rank=0,
            addresses=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
            run_id=run_id,
            local_state_dir=tmp_path / "rank-0",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"slurm": {}}, "exactly one"),
        ({"weight_broadcast": {"type": "filesystem"}}, "NCCL"),
        (
            {
                "trainer": {"rollout_transport": {"type": "filesystem"}},
                "orchestrator": {"rollout_transport": {"type": "filesystem"}},
            },
            "ZMQ",
        ),
        ({"external_cluster": {"consensus_port": 8100}}, "must not collide"),
        (
            {
                "trainer": {
                    "rollout_transport": {"type": "zmq", "port": 5555, "hwm": 10},
                    "metrics_server": {"port": 5556},
                }
            },
            "must not collide",
        ),
        (
            {
                "inference": {
                    "parallel": {"tp": 1, "dp": 2},
                    "deployment": {"type": "single_node", "gpus_per_node": 1, "backend_port": 8100},
                }
            },
            "one inference data-parallel rank",
        ),
        (
            {
                "inference": {
                    "parallel": {"tp": 1},
                    "deployment": {
                        "type": "multi_node",
                        "gpus_per_node": 1,
                        "num_nodes": 1,
                    },
                    "slurm": {},
                }
            },
            "single-node inference replica",
        ),
        (
            {
                "orchestrator": {
                    "rollout_transport": {
                        "type": "zmq",
                        "port": 5555,
                        "hwm": 10,
                    },
                    "model": {
                        "client": {
                            "elastic": {
                                "hostname": "policy.internal",
                            }
                        }
                    },
                }
            },
            "static inference client endpoints",
        ),
        ({"clean_output_dir": True}, "does not delete shared output"),
        ({"ckpt": {"resume_step": 1}}, "does not yet support checkpoint resume"),
        (
            {
                "trainer": {
                    "rollout_transport": {"type": "zmq", "port": 5555, "hwm": 10},
                    "ckpt": {"resume_step": 1},
                },
                "orchestrator": {
                    "rollout_transport": {"type": "zmq", "port": 5555, "hwm": 10},
                    "ckpt": {"resume_step": 1},
                },
            },
            "does not yet support checkpoint resume",
        ),
    ],
)
def test_external_cluster_rejects_unsupported_or_ambiguous_configs(tmp_path, override, message):
    with pytest.raises(ValidationError, match=message):
        external_config(tmp_path, **override)


def test_tensor_parallel_size_rejects_zero_before_auto_setup(tmp_path):
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        external_config(
            tmp_path,
            inference={
                "parallel": {"tp": 0},
                "deployment": {"type": "single_node", "gpus_per_node": 1, "backend_port": 8100},
            },
        )


def test_external_cluster_cli_preserves_rl_config_overrides(tmp_path):
    args, overrides = parse_args(
        [
            "--config",
            str(tmp_path / "rl.toml"),
            "--rank",
            "1",
            "--addresses",
            "10.0.0.1",
            "10.0.0.2",
            "--run-id",
            "run-1",
            "--local-state-dir",
            str(tmp_path / "rank-1"),
            "--output-dir",
            "/outputs/run-1",
            "--wandb.name",
            "run-1",
            "--wandb.group",
            "smoke",
        ]
    )

    assert args.rank == 1
    assert args.addresses == ["10.0.0.1", "10.0.0.2"]
    assert overrides == [
        "--output-dir",
        "/outputs/run-1",
        "--wandb.name",
        "run-1",
        "--wandb.group",
        "smoke",
    ]


def test_local_process_supervisor_reports_nonzero_exit(tmp_path):
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="failing",
                argv=(sys.executable, "-c", "raise SystemExit(7)"),
                env=os.environ,
                log_path=tmp_path / "failing.log",
            )
        ],
        termination_grace_period=0.1,
    )

    supervisor.start()
    result = _wait_for_local_result(supervisor)
    supervisor.terminate()

    assert result == "failing exited with code 7."


def test_local_process_supervisor_waits_for_all_components(tmp_path):
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="orchestrator",
                argv=(sys.executable, "-c", "pass"),
                env=os.environ,
                log_path=tmp_path / "orchestrator.log",
            ),
            ComponentPlan(
                name="trainer",
                argv=(sys.executable, "-c", "import time; time.sleep(0.2)"),
                env=os.environ,
                log_path=tmp_path / "trainer.log",
            ),
        ],
        termination_grace_period=0.1,
    )

    supervisor.start()
    time.sleep(0.05)
    assert supervisor.poll() is None
    assert _wait_for_local_result(supervisor) == ""
    supervisor.terminate()


def test_local_process_supervisor_fails_when_either_component_fails(tmp_path):
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="orchestrator",
                argv=(sys.executable, "-c", "raise SystemExit(9)"),
                env=os.environ,
                log_path=tmp_path / "orchestrator.log",
            ),
            ComponentPlan(
                name="trainer",
                argv=(sys.executable, "-c", "import time; time.sleep(60)"),
                env=os.environ,
                log_path=tmp_path / "trainer.log",
            ),
        ],
        termination_grace_period=0.05,
    )

    supervisor.start()
    assert _wait_for_local_result(supervisor) == "orchestrator exited with code 9."
    supervisor.terminate()
    assert supervisor.returncodes["trainer"] is not None


def test_local_process_supervisor_escalates_term_to_kill(tmp_path):
    started = tmp_path / "started"
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="stubborn",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, signal, time; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        f"pathlib.Path({str(started)!r}).write_text('ready'); "
                        "time.sleep(60)"
                    ),
                ),
                env=os.environ,
                log_path=tmp_path / "stubborn.log",
            )
        ],
        termination_grace_period=0.05,
    )

    supervisor.start()
    _wait_for_path(started)
    assert supervisor.terminate() == frozenset({"stubborn"})

    assert supervisor.returncodes["stubborn"] == -signal.SIGKILL


def test_local_process_supervisor_terminates_descendants_after_leader_exits(
    tmp_path,
):
    child_marker = tmp_path / "child-finished"
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="leader",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import subprocess, sys; "
                        "subprocess.Popen([sys.executable, '-c', "
                        f'"import pathlib, time; time.sleep(0.5); '
                        f"pathlib.Path({str(child_marker)!r}).write_text('alive')\"]);"
                    ),
                ),
                env=os.environ,
                log_path=tmp_path / "leader.log",
            )
        ],
        termination_grace_period=0.05,
    )

    supervisor.start()
    assert _wait_for_local_result(supervisor) == ""
    assert supervisor.terminate() == frozenset()
    time.sleep(0.6)

    assert not child_marker.exists()


def test_local_process_supervisor_detects_leader_exit_after_shutdown_check(
    tmp_path,
):
    leader_ready = tmp_path / "leader-ready"
    leader_exit = tmp_path / "leader-exit"
    child_marker = tmp_path / "child-finished"
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="inference",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, subprocess, sys, time; "
                        "subprocess.Popen([sys.executable, '-c', "
                        f'"import pathlib, time; time.sleep(0.5); '
                        f"pathlib.Path({str(child_marker)!r}).write_text('alive')\"]); "
                        f"pathlib.Path({str(leader_ready)!r}).write_text('ready'); "
                        f"exit_path = pathlib.Path({str(leader_exit)!r}); "
                        "exec('while not exit_path.exists():\\n time.sleep(0.01)')"
                    ),
                ),
                env=os.environ,
                log_path=tmp_path / "inference.log",
            )
        ],
        termination_grace_period=0.05,
    )

    supervisor.start()
    _wait_for_path(leader_ready)
    assert supervisor.shutdown_failure(require_running=True) is None
    leader_exit.write_text("exit")
    assert _wait_for_local_result(supervisor) == ""
    assert supervisor.terminate() == frozenset()
    time.sleep(0.6)

    assert not child_marker.exists()


def test_local_process_supervisor_detects_clean_inference_exit_before_stop(
    tmp_path,
):
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="inference",
                argv=(sys.executable, "-c", "pass"),
                env=os.environ,
                log_path=tmp_path / "inference.log",
            )
        ],
        termination_grace_period=0.05,
    )

    supervisor.start()
    assert _wait_for_local_result(supervisor) == ""

    assert supervisor.shutdown_failure(require_running=True) == ("inference exited before coordinated shutdown.")
    assert supervisor.terminate() == frozenset()


def test_local_process_supervisor_does_not_signal_retired_process_group(
    monkeypatch,
    tmp_path,
):
    supervisor = LocalProcessSupervisor(
        [
            ComponentPlan(
                name="completed",
                argv=(sys.executable, "-c", "pass"),
                env=os.environ,
                log_path=tmp_path / "completed.log",
            )
        ],
        termination_grace_period=0.05,
    )
    supervisor.start()
    assert _wait_for_local_result(supervisor) == ""
    monkeypatch.setattr(
        "prime_rl.external_cluster.os.killpg",
        lambda *_args: pytest.fail("retired process group was signaled"),
    )

    assert supervisor.terminate() == frozenset()


def test_external_cluster_dry_run_never_launches_processes(
    monkeypatch,
    tmp_path,
):
    config = external_config(tmp_path, dry_run=True)
    monkeypatch.setattr(
        "prime_rl.external_cluster.run_external_node",
        lambda *_args, **_kwargs: pytest.fail("dry run launched a cluster"),
    )

    run_external_cluster(
        config,
        rank=0,
        addresses=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
        run_id="dry-run",
        local_state_dir=tmp_path / "rank-0",
    )

    assert not config.output_dir.exists()


def test_external_cluster_dry_run_rejects_nonempty_output(tmp_path):
    config = external_config(tmp_path, dry_run=True)
    config.output_dir.mkdir()
    (config.output_dir / "stale").write_text("old run")

    with pytest.raises(FileExistsError, match="must be empty"):
        run_external_cluster(
            config,
            rank=0,
            addresses=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
            run_id="dry-run",
            local_state_dir=tmp_path / "rank-0",
        )


def test_external_cluster_rejects_nonempty_output_before_launch(tmp_path):
    plan = _local_cluster_plans(tmp_path)[0]
    plan.resolved_config.output_dir.mkdir()
    (plan.resolved_config.output_dir / "stale").write_text("old run")

    with pytest.raises(FileExistsError, match="must be empty"):
        run_external_node(plan)


def test_external_cluster_rejects_nonempty_checkpoint_output_before_launch(tmp_path):
    config = external_config(
        tmp_path,
        ckpt={"output_dir": tmp_path / "checkpoint-output"},
    )
    plan = build_external_node_plan(
        config,
        rank=0,
        addresses=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
        run_id="stale-checkpoint",
        local_state_dir=tmp_path / "rank-0",
    )
    assert plan.resolved_config.trainer.ckpt is not None
    checkpoint_output = plan.resolved_config.trainer.ckpt.output_dir
    assert checkpoint_output is not None
    checkpoint_output.mkdir()
    (checkpoint_output / "stale").write_text("old checkpoint")

    with pytest.raises(FileExistsError, match="must be empty"):
        run_external_node(plan)


def test_external_nodes_finalize_before_cluster_success(tmp_path):
    plans = _local_cluster_plans(tmp_path)
    finalized: list[int] = []
    errors = _run_local_cluster(
        plans,
        finalizers={
            0: lambda: finalized.append(0),
            1: lambda: finalized.append(1),
        },
    )

    assert errors == []
    assert sorted(finalized) == [0, 1]


def test_external_node_finalization_failure_fails_every_rank(tmp_path):
    plans = _local_cluster_plans(tmp_path)

    def fail_finalization() -> None:
        raise RuntimeError("volume commit failed")

    errors = _run_local_cluster(plans, finalizers={1: fail_finalization})

    assert len(errors) == 2
    assert all(isinstance(error, ExternalClusterError) for error in errors)
    assert all("volume commit failed" in str(error) for error in errors)


def test_external_node_cancellation_fails_every_rank(tmp_path):
    plans = _local_cluster_plans(tmp_path)
    cancelled = Event()
    plans[1] = replace(
        plans[1],
        components=(
            ComponentPlan(
                name="trainer",
                argv=(sys.executable, "-c", "import time; time.sleep(60)"),
                env=os.environ,
                log_path=tmp_path / "trainer.log",
            ),
        ),
    )
    trigger = Thread(target=lambda: (time.sleep(0.1), cancelled.set()))
    trigger.start()

    errors = _run_local_cluster(plans, cancel_events={1: cancelled})
    trigger.join()

    assert len(errors) == 2
    assert all(isinstance(error, ExternalClusterError) for error in errors)
    assert any("cancelled" in str(error) for error in errors)


def test_external_subconfig_failure_fails_every_rank(tmp_path):
    plans = _local_cluster_plans(tmp_path)
    plans[0].local_state_dir.write_text("not a directory")

    errors = _run_local_cluster(plans)

    assert len(errors) == 2
    assert any(isinstance(error, NotADirectoryError) for error in errors)
    assert any(isinstance(error, ExternalClusterError) for error in errors)


def test_cluster_consensus_waits_for_every_trainer_and_rank():
    port = _free_port()
    coordinator = ClusterCoordinator(
        host="127.0.0.1",
        port=port,
        world_size=3,
        trainer_ranks=frozenset({1, 2}),
        launch_identity={**_LAUNCH_IDENTITY, "world_size": 3},
        startup_timeout=2,
        shutdown_timeout=2,
    )
    coordinator.start()
    clients = [
        ClusterControlClient.connect(
            host="127.0.0.1",
            port=port,
            rank=rank,
            launch_identity={**_LAUNCH_IDENTITY, "world_size": 3},
            startup_timeout=2,
        )
        for rank in range(3)
    ]

    assert [client.receive(2) for client in clients] == [{"type": "start"}] * 3
    clients[1].send({"type": "trainer_done", "rank": 1})
    assert clients[0].receive(0.1) is None
    clients[2].send({"type": "trainer_done", "rank": 2})
    assert [client.receive(2) for client in clients] == [{"type": "stop"}] * 3
    for rank, client in enumerate(clients):
        client.send({"type": "stopped", "rank": rank})
    complete = [client.receive(2) for client in clients]

    assert all(message is not None and message["type"] == "complete" and message["success"] for message in complete)
    for client in clients:
        client.close()
    coordinator.wait()


def test_cluster_consensus_propagates_peer_failure():
    port = _free_port()
    coordinator = ClusterCoordinator(
        host="127.0.0.1",
        port=port,
        world_size=2,
        trainer_ranks=frozenset({1}),
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=2,
        shutdown_timeout=2,
    )
    coordinator.start()
    clients = [
        ClusterControlClient.connect(
            host="127.0.0.1",
            port=port,
            rank=rank,
            launch_identity=_LAUNCH_IDENTITY,
            startup_timeout=2,
        )
        for rank in range(2)
    ]
    assert [client.receive(2) for client in clients] == [{"type": "start"}] * 2

    clients[1].send({"type": "failed", "rank": 1, "reason": "trainer exploded"})
    assert [client.receive(2) for client in clients] == [
        {"type": "abort", "reason": "trainer exploded"},
        {"type": "abort", "reason": "trainer exploded"},
    ]
    for rank, client in enumerate(clients):
        client.send({"type": "stopped", "rank": rank})
    complete = [client.receive(2) for client in clients]

    assert all(message is not None and message["success"] is False for message in complete)
    assert all(message is not None and message["reason"] == "trainer exploded" for message in complete)
    for client in clients:
        client.close()
    coordinator.wait()


def test_cluster_consensus_treats_peer_disconnect_as_failure():
    port = _free_port()
    coordinator = ClusterCoordinator(
        host="127.0.0.1",
        port=port,
        world_size=2,
        trainer_ranks=frozenset({1}),
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=2,
        shutdown_timeout=0.2,
    )
    coordinator.start()
    inference = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=0,
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=2,
    )
    trainer = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=1,
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=2,
    )
    assert inference.receive(2) == {"type": "start"}
    assert trainer.receive(2) == {"type": "start"}

    trainer.close()
    abort = inference.receive(2)
    assert abort is not None
    assert abort["type"] == "abort"
    assert "disconnected" in abort["reason"]
    inference.send({"type": "stopped", "rank": 0})
    complete = inference.receive(2)

    assert complete is not None and complete["success"] is False
    inference.close()
    coordinator.wait()


def test_external_node_treats_lost_coordinator_connection_as_failure(tmp_path):
    plans = _local_cluster_plans(tmp_path)
    plan = replace(
        plans[1],
        components=(
            ComponentPlan(
                name="trainer",
                argv=(sys.executable, "-c", "import time; time.sleep(60)"),
                env=os.environ,
                log_path=tmp_path / "trainer.log",
            ),
        ),
    )
    external = plan.resolved_config.external_cluster
    assert external is not None
    listening = Event()

    def serve_then_disconnect() -> None:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((plan.addresses[0], external.consensus_port))
            listener.listen()
            listening.set()
            connection, _ = listener.accept()
            with connection:
                assert b'"type":"ready"' in connection.recv(1024)
                connection.sendall(b'{"type":"start"}\n')

    server = Thread(target=serve_then_disconnect)
    server.start()
    assert listening.wait(timeout=2)

    with pytest.raises(ExternalClusterError, match="connection closed"):
        run_external_node(plan)

    server.join(timeout=2)
    assert not server.is_alive()


def test_cluster_consensus_rejects_missing_rank():
    port = _free_port()
    coordinator = ClusterCoordinator(
        host="127.0.0.1",
        port=port,
        world_size=2,
        trainer_ranks=frozenset({1}),
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=0.2,
        shutdown_timeout=0.2,
    )
    coordinator.start()
    client = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=0,
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=1,
    )

    message = client.receive(1)
    assert message is not None
    assert message["type"] == "abort"
    assert "joined ranks=[0]" in message["reason"]
    client.close()
    with pytest.raises(ExternalClusterError, match="coordinator failed"):
        coordinator.wait()


def test_cluster_consensus_rejects_duplicate_rank():
    port = _free_port()
    coordinator = ClusterCoordinator(
        host="127.0.0.1",
        port=port,
        world_size=2,
        trainer_ranks=frozenset({1}),
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=1,
        shutdown_timeout=0.2,
    )
    coordinator.start()
    first = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=0,
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=1,
    )
    duplicate = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=0,
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=1,
    )

    message = first.receive(1)
    assert message is not None
    assert message["type"] == "abort"
    assert message["reason"] == "Cluster rank 0 joined more than once."
    first.close()
    duplicate.close()
    with pytest.raises(ExternalClusterError, match="coordinator failed"):
        coordinator.wait()


def test_cluster_consensus_rejects_mismatched_launch_identity():
    port = _free_port()
    coordinator = ClusterCoordinator(
        host="127.0.0.1",
        port=port,
        world_size=2,
        trainer_ranks=frozenset({1}),
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=1,
        shutdown_timeout=0.2,
    )
    coordinator.start()
    first = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=0,
        launch_identity=_LAUNCH_IDENTITY,
        startup_timeout=1,
    )
    mismatched = ClusterControlClient.connect(
        host="127.0.0.1",
        port=port,
        rank=1,
        launch_identity={
            **_LAUNCH_IDENTITY,
            "config_fingerprint": "different-config",
        },
        startup_timeout=1,
    )

    message = first.receive(1)
    assert message is not None
    assert message["type"] == "abort"
    assert "different run, topology, or configuration" in message["reason"]
    first.close()
    mismatched.close()
    with pytest.raises(ExternalClusterError, match="coordinator failed"):
        coordinator.wait()


def _wait_for_local_result(supervisor: LocalProcessSupervisor) -> str:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = supervisor.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("Local process did not exit.")


def _wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} was not created.")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _local_cluster_plans(tmp_path: Path):
    consensus_port = _free_port()
    rdzv_port = _free_port()
    weight_port = _free_port()
    while len({consensus_port, rdzv_port, weight_port, 5555, 5556, 5557}) < 6:
        weight_port = _free_port()
    config = external_config(
        tmp_path,
        deployment={
            "type": "multi_node",
            "gpus_per_node": 1,
            "num_train_nodes": 1,
            "num_infer_nodes": 1,
        },
        weight_broadcast={"type": "nccl", "port": weight_port},
        external_cluster={
            "trainer_rdzv_port": rdzv_port,
            "consensus_port": consensus_port,
            "startup_timeout": 2,
            "shutdown_timeout": 2,
            "termination_grace_period": 0.05,
        },
    )
    addresses = ["127.0.0.1", "127.0.0.2"]
    plans = [
        build_external_node_plan(
            config,
            rank=rank,
            addresses=addresses,
            run_id="local-run",
            local_state_dir=tmp_path / f"rank-{rank}",
            environ=os.environ,
        )
        for rank in range(2)
    ]
    plans[0] = replace(
        plans[0],
        components=(
            ComponentPlan(
                name="inference",
                argv=(sys.executable, "-c", "import time; time.sleep(60)"),
                env=os.environ,
                log_path=tmp_path / "inference.log",
            ),
        ),
    )
    plans[1] = replace(
        plans[1],
        components=(
            ComponentPlan(
                name="trainer",
                argv=(sys.executable, "-c", "pass"),
                env=os.environ,
                log_path=tmp_path / "trainer.log",
            ),
        ),
    )
    return plans


def _run_local_cluster(plans, *, finalizers=None, cancel_events=None):
    finalizers = finalizers or {}
    cancel_events = cancel_events or {}
    errors: list[BaseException] = []

    def run(plan) -> None:
        try:
            run_external_node(
                plan,
                finalize=finalizers.get(plan.rank),
                cancel_event=cancel_events.get(plan.rank),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=run, args=(plan,)) for plan in plans]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    return errors
