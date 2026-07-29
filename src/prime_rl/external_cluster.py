"""Provider-neutral execution for externally allocated PrimeRL clusters.

Providers may inspect ``build_external_node_plan`` or call ``run_external_cluster``
in process. The equivalent command is:

``external-cluster --config RL_TOML --rank R --addresses HOST... --run-id ID
--local-state-dir DIR [RLConfig overrides...]``
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import IO, Any, Callable, Mapping, Sequence

import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import to_toml_dict
from prime_rl.utils.logger import get_logger
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_INFERENCE_ENV_VARS,
    DEFAULT_TRAINER_ENV_VARS,
)

EXTERNAL_CLUSTER_COMMAND = "external-cluster"
TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
_MAX_CONTROL_MESSAGE_BYTES = 65536


class NodeRole(StrEnum):
    INFERENCE = "inference"
    TRAINER = "trainer"


class ExternalClusterError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentPlan:
    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    log_path: Path


@dataclass(frozen=True)
class ExternalNodePlan:
    rank: int
    role: NodeRole
    trainer_node_rank: int | None
    first_trainer_rank: int
    first_trainer_address: str
    addresses: tuple[str, ...]
    run_id: str
    local_state_dir: Path
    components: tuple[ComponentPlan, ...]
    resolved_config: RLConfig
    address_fingerprint: str
    config_fingerprint: str

    @property
    def trainer_ranks(self) -> frozenset[int]:
        return frozenset(range(self.first_trainer_rank, len(self.addresses)))

    @property
    def launch_identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "world_size": len(self.addresses),
            "address_fingerprint": self.address_fingerprint,
            "config_fingerprint": self.config_fingerprint,
        }


def build_external_node_plan(
    config: RLConfig,
    *,
    rank: int,
    addresses: Sequence[str],
    run_id: str,
    local_state_dir: Path,
    environ: Mapping[str, str] | None = None,
    launcher_argv: Sequence[str] | None = None,
) -> ExternalNodePlan:
    """Resolve the provider-neutral process plan for one externally allocated node.

    ``addresses`` is provider-rank ordered and ``local_state_dir`` must be private
    to this rank. The caller owns allocation and durable storage; Prime owns role
    mapping, runtime endpoint injection, subconfigs, and child commands.
    """

    resolved = config.model_copy(deep=True)
    external = resolved.external_cluster
    if resolved.deployment.type != "multi_node" or external is None:
        raise ValueError("External node planning requires a validated external_cluster multi-node config.")

    ordered_addresses = tuple(_normalize_address(address) for address in addresses)
    expected_nodes = resolved.deployment.total_infer_nodes + resolved.deployment.num_train_nodes
    if len(ordered_addresses) != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} provider addresses, got {len(ordered_addresses)}.")
    if len(set(ordered_addresses)) != len(ordered_addresses):
        raise ValueError("Provider addresses must be unique and ordered by provider rank.")
    if not 0 <= rank < expected_nodes:
        raise ValueError(f"Provider rank {rank} is outside [0, {expected_nodes}).")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9_.-]*.")
    local_state_dir = local_state_dir.expanduser()
    if not local_state_dir.is_absolute():
        raise ValueError("local_state_dir must be an absolute rank-local path.")
    address_fingerprint = _fingerprint(ordered_addresses)
    config_fingerprint = _fingerprint(resolved.model_dump(mode="json"))

    first_trainer_rank = resolved.deployment.total_infer_nodes
    first_trainer_address = ordered_addresses[first_trainer_rank]
    role = NodeRole.INFERENCE if rank < first_trainer_rank else NodeRole.TRAINER
    trainer_node_rank = rank - first_trainer_rank if role is NodeRole.TRAINER else None

    assert resolved.inference is not None
    assert resolved.trainer.rollout_transport.type == "zmq"
    assert resolved.orchestrator.rollout_transport.type == "zmq"
    assert resolved.orchestrator.weight_broadcast.type == "nccl"
    resolved.trainer.rollout_transport.host = _zmq_host(first_trainer_address)
    resolved.orchestrator.rollout_transport.host = _zmq_host(first_trainer_address)
    resolved.orchestrator.weight_broadcast.host = first_trainer_address

    inference_port = resolved.inference.deployment.backend_port
    inference_urls = [_http_url(address, inference_port, "/v1") for address in ordered_addresses[:first_trainer_rank]]
    resolved.orchestrator.model.client.base_url = inference_urls
    resolved.orchestrator.model.client.admin_base_url = inference_urls
    resolved.orchestrator.model.client.dp_rank_count = 1

    resolved.inference.server.host = ordered_addresses[rank] if role is NodeRole.INFERENCE else None
    resolved.inference.server.port = inference_port
    resolved.inference.parallel.dp = 1
    resolved.inference.data_parallel_size_local = 1
    resolved.inference.api_server_count = 1

    config_dir = local_state_dir / "configs"
    log_root = resolved.output_dir / "logs"
    base_env = dict(os.environ if environ is None else environ)
    visible_devices = _visible_devices(base_env, resolved.deployment.gpus_per_node)
    observed_launcher_argv = list(
        launcher_argv or _launcher_argv(config, rank, ordered_addresses, run_id, local_state_dir)
    )
    shared_wandb_env = {
        "WANDB_SHARED_MODE": "1",
        "WANDB_SHARED_RUN_ID": base_env.get("WANDB_SHARED_RUN_ID", run_id),
        "WANDB_PROGRAM": EXTERNAL_CLUSTER_COMMAND,
        "WANDB_ARGS": json.dumps(observed_launcher_argv),
    }
    components: list[ComponentPlan] = []
    if role is NodeRole.INFERENCE:
        components.append(
            ComponentPlan(
                name="inference",
                argv=("inference", "@", str(config_dir / INFERENCE_TOML)),
                env={
                    **base_env,
                    **DEFAULT_COMMON_ENV_VARS,
                    **DEFAULT_INFERENCE_ENV_VARS,
                    **resolved.env_vars,
                    **resolved.inference.env_vars,
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                },
                log_path=log_root / "inference" / f"node_{rank}.log",
            )
        )
    else:
        assert trainer_node_rank is not None
        if trainer_node_rank == 0:
            components.append(
                ComponentPlan(
                    name="orchestrator",
                    argv=("orchestrator", "@", str(config_dir / ORCHESTRATOR_TOML)),
                    env={
                        **base_env,
                        **DEFAULT_COMMON_ENV_VARS,
                        **resolved.env_vars,
                        **resolved.orchestrator.env_vars,
                        **shared_wandb_env,
                        "WANDB_SHARED_LABEL": "orchestrator",
                        "LOGURU_FORCE_COLORS": "1",
                    },
                    log_path=log_root / "orchestrator.log",
                )
            )

        components.append(
            ComponentPlan(
                name="trainer",
                argv=(
                    "torchrun",
                    "--role=trainer",
                    f"--nnodes={resolved.deployment.num_train_nodes}",
                    f"--nproc-per-node={resolved.deployment.gpus_per_node}",
                    f"--node-rank={trainer_node_rank}",
                    f"--rdzv-endpoint={_host_port(first_trainer_address, external.trainer_rdzv_port)}",
                    f"--rdzv-id={run_id}",
                    f"--log-dir={log_root / 'trainer' / 'torchrun'}",
                    "--tee=3",
                    "--redirects=3",
                    f"--local-ranks-filter={','.join(map(str, resolved.trainer.log.ranks_filter))}",
                    "-m",
                    "prime_rl.trainer.rl.train",
                    "@",
                    str(config_dir / TRAINER_TOML),
                ),
                env={
                    **base_env,
                    **DEFAULT_COMMON_ENV_VARS,
                    **DEFAULT_TRAINER_ENV_VARS,
                    **resolved.env_vars,
                    **resolved.trainer.env_vars,
                    **shared_wandb_env,
                    "WANDB_SHARED_LABEL": "trainer",
                    "LOGURU_FORCE_COLORS": "1",
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                },
                log_path=log_root / "trainer" / f"node_{trainer_node_rank}.log",
            )
        )

    return ExternalNodePlan(
        rank=rank,
        role=role,
        trainer_node_rank=trainer_node_rank,
        first_trainer_rank=first_trainer_rank,
        first_trainer_address=first_trainer_address,
        addresses=ordered_addresses,
        run_id=run_id,
        local_state_dir=local_state_dir,
        components=tuple(components),
        resolved_config=resolved,
        address_fingerprint=address_fingerprint,
        config_fingerprint=config_fingerprint,
    )


def write_external_subconfigs(plan: ExternalNodePlan) -> None:
    """Materialize launch configs locally and one-writer evidence durably."""

    config_dir = plan.local_state_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_toml(config_dir / TRAINER_TOML, to_toml_dict(plan.resolved_config.trainer))
    _write_toml(config_dir / ORCHESTRATOR_TOML, to_toml_dict(plan.resolved_config.orchestrator))
    if plan.role is NodeRole.INFERENCE:
        assert plan.resolved_config.inference is not None
        _write_toml(
            config_dir / INFERENCE_TOML,
            to_toml_dict(
                plan.resolved_config.inference,
                exclude={"deployment", "slurm", "output_dir", "dry_run"},
            ),
        )

    evidence_dir = plan.resolved_config.output_dir / "configs"
    if plan.rank == 0:
        assert plan.resolved_config.inference is not None
        _write_toml(
            evidence_dir / INFERENCE_TOML,
            to_toml_dict(
                plan.resolved_config.inference,
                exclude={"deployment", "slurm", "output_dir", "dry_run"},
            ),
        )
    if plan.trainer_node_rank == 0:
        _write_toml(evidence_dir / TRAINER_TOML, to_toml_dict(plan.resolved_config.trainer))
        _write_toml(evidence_dir / ORCHESTRATOR_TOML, to_toml_dict(plan.resolved_config.orchestrator))


def run_external_cluster(
    config: RLConfig,
    *,
    rank: int,
    addresses: Sequence[str],
    run_id: str,
    local_state_dir: Path,
    environ: Mapping[str, str] | None = None,
    launcher_argv: Sequence[str] | None = None,
    cancel_event: Event | None = None,
    finalize: Callable[[], None] | None = None,
) -> None:
    """Build and run one external node.

    A provider may pass a synchronous ``finalize`` callback to publish this
    rank's durable artifacts. It runs after local Prime processes stop cleanly
    and before this rank acknowledges terminal success, so publication failure
    participates in the same all-rank consensus.
    """

    plan = build_external_node_plan(
        config,
        rank=rank,
        addresses=addresses,
        run_id=run_id,
        local_state_dir=local_state_dir,
        environ=environ,
        launcher_argv=launcher_argv,
    )
    if config.dry_run:
        _validate_fresh_output_dirs(plan.resolved_config)
        return
    run_external_node(plan, cancel_event=cancel_event, finalize=finalize)


def run_external_node(
    plan: ExternalNodePlan,
    *,
    cancel_event: Event | None = None,
    finalize: Callable[[], None] | None = None,
) -> None:
    """Run one external node until cluster-wide terminal consensus."""

    external = plan.resolved_config.external_cluster
    assert external is not None
    cancel_event = cancel_event or Event()
    coordinator: ClusterCoordinator | None = None
    client: ClusterControlClient | None = None
    supervisor = LocalProcessSupervisor(plan.components, external.termination_grace_period)
    primary_error: BaseException | None = None
    sent_terminal = False
    try:
        _validate_fresh_output_dirs(plan.resolved_config)
        if plan.rank == 0:
            coordinator = ClusterCoordinator(
                host=plan.addresses[0],
                port=external.consensus_port,
                world_size=len(plan.addresses),
                trainer_ranks=plan.trainer_ranks,
                launch_identity=plan.launch_identity,
                startup_timeout=external.startup_timeout,
                shutdown_timeout=external.shutdown_timeout,
            )
            coordinator.start()

        client = ClusterControlClient.connect(
            host=plan.addresses[0],
            port=external.consensus_port,
            rank=plan.rank,
            launch_identity=plan.launch_identity,
            startup_timeout=external.startup_timeout,
            cancel_event=cancel_event,
        )
        _wait_for_cluster_start(client, cancel_event, external.startup_timeout, plan.rank)
        write_external_subconfigs(plan)
        supervisor.start()
        trainer_done = False
        while True:
            if cancel_event.is_set():
                raise ExternalClusterError(f"External node {plan.rank} was cancelled.")

            local_state = supervisor.poll()
            if local_state is not None and not trainer_done:
                if local_state:
                    client.send({"type": "failed", "rank": plan.rank, "reason": local_state})
                    raise ExternalClusterError(local_state)
                if plan.role is NodeRole.INFERENCE:
                    reason = f"Inference process on rank {plan.rank} exited before coordinated shutdown."
                    client.send({"type": "failed", "rank": plan.rank, "reason": reason})
                    raise ExternalClusterError(reason)
                client.send({"type": "trainer_done", "rank": plan.rank})
                trainer_done = True

            command = client.receive(0.1)
            if command is None:
                continue
            command_type = command.get("type")
            if command_type == "stop":
                shutdown_failure = supervisor.shutdown_failure(
                    require_running=plan.role is NodeRole.INFERENCE,
                )
                if plan.role is NodeRole.TRAINER and not trainer_done:
                    shutdown_failure = shutdown_failure or (
                        f"Trainer rank {plan.rank} received shutdown before its components completed."
                    )
                signaled = supervisor.terminate()
                if shutdown_failure is None and plan.role is NodeRole.INFERENCE and "inference" not in signaled:
                    shutdown_failure = "inference exited before coordinated shutdown."
                if shutdown_failure is not None:
                    primary_error = ExternalClusterError(shutdown_failure)
                    client.send(
                        {
                            "type": "failed",
                            "rank": plan.rank,
                            "reason": shutdown_failure,
                        }
                    )
                if primary_error is None:
                    try:
                        if finalize is not None:
                            finalize()
                    except BaseException as finalize_error:
                        primary_error = ExternalClusterError(
                            f"Rank {plan.rank} artifact finalization failed: {finalize_error}"
                        )
                        client.send(
                            {
                                "type": "failed",
                                "rank": plan.rank,
                                "reason": str(primary_error),
                            }
                        )
                client.send({"type": "stopped", "rank": plan.rank})
                sent_terminal = True
            elif command_type == "abort":
                reason = str(command.get("reason", "A cluster peer failed."))
                supervisor.terminate()
                client.send({"type": "stopped", "rank": plan.rank})
                sent_terminal = True
                primary_error = ExternalClusterError(reason)
            elif command_type == "complete":
                if not command.get("success", False):
                    raise ExternalClusterError(str(command.get("reason", "Cluster run failed.")))
                return
            else:
                raise ExternalClusterError(f"Unexpected cluster-control message: {command}.")

            if sent_terminal:
                complete = client.receive(external.shutdown_timeout)
                if complete is None:
                    raise ExternalClusterError("Timed out waiting for cluster completion.")
                if complete.get("type") != "complete":
                    raise ExternalClusterError(f"Expected cluster completion, got {complete}.")
                if not complete.get("success", False):
                    raise primary_error or ExternalClusterError(str(complete.get("reason", "Cluster run failed.")))
                return
    except BaseException as exc:
        primary_error = primary_error or exc
        if client is not None and not sent_terminal:
            try:
                client.send({"type": "failed", "rank": plan.rank, "reason": str(primary_error)})
            except Exception:
                pass
            try:
                supervisor.terminate()
            except Exception as cleanup_error:
                get_logger().error(f"Failed to terminate local Prime processes: {cleanup_error}")
            try:
                client.send({"type": "stopped", "rank": plan.rank})
            except Exception:
                pass
        elif coordinator is not None:
            coordinator.cancel(str(primary_error))
        raise primary_error
    finally:
        if client is not None:
            client.close()
        if coordinator is not None:
            if primary_error is not None:
                coordinator.cancel(str(primary_error))
            try:
                coordinator.wait()
            except BaseException as coordinator_error:
                if primary_error is None:
                    raise
                get_logger().error(f"Cluster coordinator cleanup failed: {coordinator_error}")


class LocalProcessSupervisor:
    def __init__(self, plans: Sequence[ComponentPlan], termination_grace_period: float):
        self._plans = tuple(plans)
        self._termination_grace_period = termination_grace_period
        self._processes: list[_ManagedProcess] = []

    def start(self) -> None:
        try:
            for plan in self._plans:
                plan.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = plan.log_path.open("w")
                try:
                    process = subprocess.Popen(
                        plan.argv,
                        env=dict(plan.env),
                        stdout=log_file,
                        stderr=log_file,
                        start_new_session=True,
                    )
                except BaseException:
                    log_file.close()
                    raise
                self._processes.append(
                    _ManagedProcess(
                        plan=plan,
                        process=process,
                        process_group_id=process.pid,
                        log_file=log_file,
                    )
                )
        except BaseException:
            try:
                self.terminate()
            except BaseException as cleanup_error:
                get_logger().error(f"Failed to clean up after component launch failure: {cleanup_error}")
            raise

    def poll(self) -> str | None:
        if not self._processes:
            return "No Prime components were launched."
        running = False
        for managed in self._processes:
            returncode = managed.process.poll()
            if returncode is None:
                running = True
            else:
                _process_group_exists(managed)
                if returncode != 0:
                    return f"{managed.plan.name} exited with code {returncode}."
        return None if running else ""

    @property
    def returncodes(self) -> dict[str, int | None]:
        return {managed.plan.name: managed.process.poll() for managed in self._processes}

    def shutdown_failure(self, *, require_running: bool) -> str | None:
        for managed in self._processes:
            returncode = managed.process.poll()
            if returncode is None:
                continue
            _process_group_exists(managed)
            if returncode != 0:
                return f"{managed.plan.name} exited with code {returncode} before coordinated shutdown."
            if require_running:
                return f"{managed.plan.name} exited before coordinated shutdown."
        return None

    def terminate(self) -> frozenset[str]:
        """Terminate local groups and report leaders alive when shutdown began."""

        active = list(self._processes)
        running_leaders = {managed.plan.name for managed in active if managed.process.poll() is None}
        for managed in active:
            _signal_process_group(managed, signal.SIGTERM)

        deadline = time.monotonic() + self._termination_grace_period
        while active and time.monotonic() < deadline:
            active = [managed for managed in active if _process_group_exists(managed)]
            if active:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        for managed in active:
            _signal_process_group(managed, signal.SIGKILL)
        cleanup_errors: list[BaseException] = []
        for managed in self._processes:
            try:
                managed.process.wait(timeout=5)
            except BaseException as exc:
                cleanup_errors.append(exc)
            finally:
                managed.log_file.close()
        if cleanup_errors:
            raise cleanup_errors[0]
        return frozenset(running_leaders)


class ClusterControlClient:
    def __init__(self, connection: _JsonLineConnection):
        self._connection = connection

    @classmethod
    def connect(
        cls,
        *,
        host: str,
        port: int,
        rank: int,
        launch_identity: Mapping[str, Any],
        startup_timeout: float,
        cancel_event: Event | None = None,
    ) -> ClusterControlClient:
        deadline = time.monotonic() + startup_timeout
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise ExternalClusterError(f"External node {rank} was cancelled before joining the cluster.")
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection((host, port), timeout=min(1.0, startup_timeout))
                connection = _JsonLineConnection(sock)
                connection.send(
                    {
                        "type": "ready",
                        "rank": rank,
                        **launch_identity,
                    }
                )
                return cls(connection)
            except OSError as exc:
                last_error = exc
                if sock is not None:
                    sock.close()
                time.sleep(0.05)
        raise ExternalClusterError(
            f"Could not connect to cluster coordinator at {_host_port(host, port)}."
        ) from last_error

    def send(self, message: Mapping[str, Any]) -> None:
        self._connection.send(message)

    def receive(self, timeout: float) -> dict[str, Any] | None:
        return self._connection.receive(timeout)

    def close(self) -> None:
        self._connection.close()


class ClusterCoordinator:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        world_size: int,
        trainer_ranks: frozenset[int],
        launch_identity: Mapping[str, Any],
        startup_timeout: float,
        shutdown_timeout: float,
    ):
        self._host = host
        self._port = port
        self._world_size = world_size
        self._trainer_ranks = trainer_ranks
        self._launch_identity = dict(launch_identity)
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._listening = Event()
        self._thread = Thread(target=self._run, name="prime-cluster-coordinator", daemon=True)
        self._cancel = Event()
        self._cancel_reason = "Cluster coordinator was cancelled."
        self._error: BaseException | None = None
        self._readiness_error: str | None = None

    def start(self) -> None:
        self._thread.start()
        if not self._listening.wait(timeout=min(10.0, self._startup_timeout)):
            raise ExternalClusterError("Cluster coordinator did not start listening.")
        if self._error is not None:
            raise ExternalClusterError("Cluster coordinator failed to start.") from self._error

    def wait(self) -> None:
        self._thread.join(timeout=self._shutdown_timeout + 5)
        if self._thread.is_alive():
            raise ExternalClusterError("Cluster coordinator did not terminate.")
        if self._error is not None:
            raise ExternalClusterError("Cluster coordinator failed.") from self._error

    def cancel(self, reason: str) -> None:
        self._cancel_reason = reason
        self._cancel.set()

    def _run(self) -> None:
        listener: socket.socket | None = None
        connections: dict[int, _JsonLineConnection] = {}
        accepted_connections: set[_JsonLineConnection] = set()
        event_queue: Queue[tuple[str, _JsonLineConnection, dict[str, Any] | None]] = Queue()
        reader_threads: list[Thread] = []
        try:
            listener = _listen(self._host, self._port)
            listener.settimeout(0.1)
            self._listening.set()
            deadline = time.monotonic() + self._startup_timeout

            while len(connections) < self._world_size and time.monotonic() < deadline and not self._cancel.is_set():
                try:
                    sock, _ = listener.accept()
                except TimeoutError:
                    pass
                else:
                    connection = _JsonLineConnection(sock)
                    accepted_connections.add(connection)
                    thread = Thread(
                        target=_read_control_messages,
                        args=(connection, event_queue),
                        name="prime-cluster-peer",
                        daemon=True,
                    )
                    thread.start()
                    reader_threads.append(thread)
                self._drain_ready_events(event_queue, connections)
                if self._readiness_error is not None:
                    break

            self._drain_ready_events(event_queue, connections)
            if self._cancel.is_set():
                self._readiness_error = self._readiness_error or self._cancel_reason
            expected_ranks = set(range(self._world_size))
            if self._readiness_error is not None:
                _broadcast(connections, {"type": "abort", "reason": self._readiness_error})
                raise ExternalClusterError(self._readiness_error)
            if set(connections) != expected_ranks:
                reason = f"Cluster readiness timed out; joined ranks={sorted(connections)}, expected={sorted(expected_ranks)}."
                _broadcast(connections, {"type": "abort", "reason": reason})
                raise ExternalClusterError(reason)

            _broadcast(connections, {"type": "start"})
            trainer_done: set[int] = set()
            stopped: set[int] = set()
            failure_reason: str | None = None
            shutdown_deadline: float | None = None
            stop_sent = False

            while True:
                if self._cancel.is_set() and failure_reason is None:
                    failure_reason = self._cancel_reason
                if failure_reason is not None and not stop_sent:
                    _broadcast(connections, {"type": "abort", "reason": failure_reason})
                    stop_sent = True
                    shutdown_deadline = time.monotonic() + self._shutdown_timeout
                if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                    failure_reason = failure_reason or (
                        f"Cluster shutdown timed out; stopped ranks={sorted(stopped)}, "
                        f"expected={list(range(self._world_size))}."
                    )
                    _broadcast(
                        connections,
                        {"type": "complete", "success": False, "reason": failure_reason},
                    )
                    return

                try:
                    kind, connection, message = event_queue.get(timeout=0.1)
                except Empty:
                    continue
                rank = _connection_rank(connections, connection)
                if rank is None:
                    if kind == "message" and message is not None and message.get("type") == "ready":
                        failure_reason = (
                            failure_reason or f"Duplicate or late ready message for rank {message.get('rank')}."
                        )
                    continue
                if kind == "disconnect":
                    if rank not in stopped:
                        failure_reason = failure_reason or f"Cluster rank {rank} disconnected."
                elif message is not None:
                    message_type = message.get("type")
                    if message.get("rank") != rank:
                        failure_reason = failure_reason or f"Cluster rank {rank} sent a mismatched rank field."
                    elif message_type == "trainer_done":
                        if rank not in self._trainer_ranks:
                            failure_reason = failure_reason or f"Inference rank {rank} reported trainer completion."
                        trainer_done.add(rank)
                    elif message_type == "failed":
                        failure_reason = failure_reason or str(message.get("reason", f"Cluster rank {rank} failed."))
                    elif message_type == "stopped":
                        if not stop_sent:
                            failure_reason = failure_reason or f"Cluster rank {rank} stopped before shutdown."
                        stopped.add(rank)
                    else:
                        failure_reason = failure_reason or f"Cluster rank {rank} sent unexpected message {message}."

                if failure_reason is not None and not stop_sent:
                    _broadcast(connections, {"type": "abort", "reason": failure_reason})
                    stop_sent = True
                    shutdown_deadline = time.monotonic() + self._shutdown_timeout
                elif trainer_done == self._trainer_ranks and not stop_sent:
                    _broadcast(connections, {"type": "stop"})
                    stop_sent = True
                    shutdown_deadline = time.monotonic() + self._shutdown_timeout

                if stop_sent and len(stopped) == self._world_size:
                    _broadcast(
                        connections,
                        {"type": "complete", "success": failure_reason is None, "reason": failure_reason},
                    )
                    return
        except BaseException as exc:
            self._error = exc
        finally:
            self._listening.set()
            for connection in connections.values():
                connection.close()
            for connection in accepted_connections:
                connection.close()
            if listener is not None:
                listener.close()
            for thread in reader_threads:
                thread.join(timeout=1)

    def _drain_ready_events(
        self,
        event_queue: Queue[tuple[str, _JsonLineConnection, dict[str, Any] | None]],
        connections: dict[int, _JsonLineConnection],
    ) -> None:
        while True:
            try:
                kind, connection, message = event_queue.get_nowait()
            except Empty:
                return
            if kind == "disconnect":
                self._readiness_error = (
                    self._readiness_error or "A cluster peer sent an invalid message before startup."
                )
                connection.close()
                continue
            if message is None:
                self._readiness_error = (
                    self._readiness_error or "A cluster peer sent an invalid message before startup."
                )
                connection.close()
                continue
            if message.get("type") == "failed":
                self._readiness_error = self._readiness_error or str(
                    message.get("reason", "A cluster peer failed before startup.")
                )
                continue
            if message.get("type") != "ready":
                self._readiness_error = (
                    self._readiness_error or "A cluster peer sent an invalid message before startup."
                )
                connection.close()
                continue
            identity = {key: message.get(key) for key in self._launch_identity}
            if identity != self._launch_identity:
                self._readiness_error = self._readiness_error or (
                    "A cluster peer joined with a different run, topology, or configuration."
                )
                connection.close()
                continue
            rank = message.get("rank")
            if not isinstance(rank, int) or not 0 <= rank < self._world_size:
                self._readiness_error = self._readiness_error or f"Cluster peer announced invalid rank {rank!r}."
                connection.close()
                continue
            if rank in connections:
                self._readiness_error = self._readiness_error or f"Cluster rank {rank} joined more than once."
                connection.close()
                continue
            connections[rank] = connection


@dataclass
class _ManagedProcess:
    plan: ComponentPlan
    process: subprocess.Popen
    process_group_id: int
    log_file: IO[str]
    process_group_retired: bool = False


class _JsonLineConnection:
    def __init__(self, sock: socket.socket):
        self._socket = sock
        self._buffer = bytearray()
        self._send_lock = Lock()

    def send(self, message: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(message), separators=(",", ":")).encode() + b"\n"
        if len(payload) > _MAX_CONTROL_MESSAGE_BYTES:
            raise ExternalClusterError("Cluster-control message exceeds the size limit.")
        with self._send_lock:
            self._socket.sendall(payload)

    def receive(self, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                payload = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                message = json.loads(payload)
                if not isinstance(message, dict):
                    raise ExternalClusterError("Cluster-control message must be a JSON object.")
                return message

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._socket.settimeout(remaining)
            try:
                chunk = self._socket.recv(65536)
            except TimeoutError:
                return None
            if not chunk:
                raise ExternalClusterError("Cluster-control connection closed.")
            self._buffer.extend(chunk)
            if len(self._buffer) > _MAX_CONTROL_MESSAGE_BYTES:
                raise ExternalClusterError("Cluster-control message exceeds the size limit.")

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()


def _read_control_messages(
    connection: _JsonLineConnection,
    event_queue: Queue[tuple[str, _JsonLineConnection, dict[str, Any] | None]],
) -> None:
    try:
        while True:
            message = connection.receive(1.0)
            if message is not None:
                event_queue.put(("message", connection, message))
    except BaseException:
        event_queue.put(("disconnect", connection, None))


def _wait_for_cluster_start(
    client: ClusterControlClient,
    cancel_event: Event,
    startup_timeout: float,
    rank: int,
) -> None:
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if cancel_event.is_set():
            raise ExternalClusterError(f"External node {rank} was cancelled before cluster start.")
        command = client.receive(min(0.1, max(0.0, deadline - time.monotonic())))
        if command is None:
            continue
        if command.get("type") != "start":
            raise ExternalClusterError(str(command.get("reason", "Cluster start was aborted.")))
        return
    raise ExternalClusterError("Timed out waiting for cluster start.")


def _connection_rank(
    connections: Mapping[int, _JsonLineConnection],
    connection: _JsonLineConnection,
) -> int | None:
    return next((rank for rank, candidate in connections.items() if candidate is connection), None)


def _broadcast(connections: Mapping[int, _JsonLineConnection], message: Mapping[str, Any]) -> None:
    for connection in connections.values():
        try:
            connection.send(message)
        except OSError:
            pass


def _listen(host: str, port: int) -> socket.socket:
    last_error: OSError | None = None
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            sock.listen()
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise ExternalClusterError(f"Could not bind cluster coordinator at {_host_port(host, port)}.") from last_error


def _signal_process_group(
    managed: _ManagedProcess,
    sig: signal.Signals,
) -> bool:
    if managed.process_group_retired:
        return False
    try:
        os.killpg(managed.process_group_id, sig)
    except ProcessLookupError:
        managed.process_group_retired = True
        return False
    except PermissionError:
        if managed.process.poll() is None:
            managed.process.send_signal(sig)
            return True
        managed.process_group_retired = True
        return False
    return True


def _process_group_exists(managed: _ManagedProcess) -> bool:
    if managed.process_group_retired:
        return False
    try:
        os.killpg(managed.process_group_id, 0)
    except ProcessLookupError:
        managed.process_group_retired = True
        return False
    except PermissionError:
        return managed.process.poll() is None
    return True


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("wb") as file:
        tomli_w.dump(data, file)
    temp_path.replace(path)


def _validate_fresh_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"External cluster output path is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"External cluster output directory must be empty; use a unique output_dir: {output_dir}")


def _validate_fresh_output_dirs(config: RLConfig) -> None:
    output_dirs = {config.output_dir}
    if config.trainer.ckpt is not None and config.trainer.ckpt.output_dir is not None:
        output_dirs.add(config.trainer.ckpt.output_dir)
    for output_dir in output_dirs:
        _validate_fresh_output_dir(output_dir)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_address(address: str) -> str:
    value = address.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        raise ValueError("Provider addresses must not be empty.")
    try:
        return str(ipaddress.IPv4Address(value))
    except ipaddress.AddressValueError as exc:
        raise ValueError(
            "external_cluster currently requires literal IPv4 provider addresses because "
            "the NCCL weight-transfer listener is IPv4-only."
        ) from exc


def _host_port(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _http_url(host: str, port: int, path: str) -> str:
    return f"http://{_host_port(host, port)}{path}"


def _zmq_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _launcher_argv(
    config: RLConfig,
    rank: int,
    addresses: Sequence[str],
    run_id: str,
    local_state_dir: Path,
) -> list[str]:
    return [
        EXTERNAL_CLUSTER_COMMAND,
        "--rank",
        str(rank),
        "--addresses",
        *addresses,
        "--run-id",
        run_id,
        "--local-state-dir",
        str(local_state_dir),
        "--output-dir",
        str(config.output_dir),
    ]


def _visible_devices(environ: Mapping[str, str], gpus_per_node: int) -> str:
    inherited = environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return ",".join(str(index) for index in range(gpus_per_node))

    devices = [device.strip() for device in inherited.split(",") if device.strip()]
    if len(devices) != gpus_per_node or devices == ["-1"]:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must expose exactly "
            f"{gpus_per_node} devices for the external cluster; got {inherited!r}."
        )
    return inherited
