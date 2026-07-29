import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import (
    prepare_rl_run,
    regular_multi_node_template_context,
    write_subconfigs,
)
from prime_rl.utils.config import cli, find_package_resource
from prime_rl.utils.process import set_proc_title

_CLUSTER_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_HOST_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,254}")
_PREPARATION_TOKEN_PREFIX = "PRIME_RL_PREPARATION_TOKEN="


@dataclass(frozen=True)
class AllocatedCluster:
    rank: int
    node_count: int
    addresses: tuple[str, ...]
    local_address: str
    cluster_id: str
    pool_namespace: str

    @classmethod
    def from_environment(cls) -> "AllocatedCluster":
        try:
            rank = int(os.environ["PRIME_RL_NODE_RANK"])
            node_count = int(os.environ["PRIME_RL_NODE_COUNT"])
            addresses_value = json.loads(os.environ["PRIME_RL_NODE_ADDRESSES_JSON"])
            local_address = os.environ["PRIME_RL_LOCAL_ADDRESS"]
            cluster_id = os.environ["PRIME_RL_CLUSTER_ID"]
        except KeyError as error:
            raise ValueError(f"Missing allocated-cluster variable: {error.args[0]}") from error
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid allocated-cluster environment: {error}") from error

        if not isinstance(addresses_value, list) or not all(isinstance(address, str) for address in addresses_value):
            raise ValueError("PRIME_RL_NODE_ADDRESSES_JSON must be a JSON array of strings.")
        addresses = tuple(addresses_value)
        pool_namespace = os.environ.get("PRIME_RL_POOL_NAMESPACE", "allocated")
        cluster = cls(
            rank=rank,
            node_count=node_count,
            addresses=addresses,
            local_address=local_address,
            cluster_id=cluster_id,
            pool_namespace=pool_namespace,
        )
        cluster.validate()
        return cluster

    def validate(self) -> None:
        if self.node_count < 1:
            raise ValueError("PRIME_RL_NODE_COUNT must be positive.")
        if len(self.addresses) != self.node_count:
            raise ValueError(f"Expected {self.node_count} node addresses, received {len(self.addresses)}.")
        if len(set(self.addresses)) != len(self.addresses):
            raise ValueError("Allocated-cluster addresses must be unique.")
        if not 0 <= self.rank < self.node_count:
            raise ValueError(f"Node rank {self.rank} is outside world size {self.node_count}.")
        if self.local_address != self.addresses[self.rank]:
            raise ValueError("PRIME_RL_LOCAL_ADDRESS must equal the address at PRIME_RL_NODE_RANK.")
        if not all(_HOST_PATTERN.fullmatch(address) for address in self.addresses):
            raise ValueError("Allocated-cluster addresses contain an unsupported character.")
        if not _CLUSTER_ID_PATTERN.fullmatch(self.cluster_id):
            raise ValueError("PRIME_RL_CLUSTER_ID must be a safe nonempty path token.")
        if not _CLUSTER_ID_PATTERN.fullmatch(self.pool_namespace):
            raise ValueError("PRIME_RL_POOL_NAMESPACE must be a safe nonempty token.")


def prepare(config: RLConfig, cluster: AllocatedCluster) -> str:
    _validate_config(config, cluster)
    if cluster.rank != 0:
        raise ValueError("Allocated-cluster preparation must run on rank 0.")
    with chdir(_project_dir()):
        prepare_rl_run(config)
        _ensure_run_directories(config)
        inference_logs = config.output_dir / "logs" / "inference"
        for log in inference_logs.glob("*.log"):
            log.unlink()
        trainer_log = config.output_dir / "logs" / "trainer.log"
        inference_log = config.output_dir / "logs" / "inference.log"
        trainer_log.unlink(missing_ok=True)
        inference_log.unlink(missing_ok=True)
        trainer_log.symlink_to("trainer/node_0.log", target_is_directory=False)
        inference_log.symlink_to("inference/node_0.log", target_is_directory=False)
        return _preparation_token(config, cluster)


def run_node(config: RLConfig, cluster: AllocatedCluster) -> int:
    project_dir = _project_dir()
    with chdir(project_dir):
        _validate_config(config, cluster)
        _require_preparation_token(config, cluster)
        templates_dir = find_package_resource("templates")
        if templates_dir is None:
            raise RuntimeError("PrimeRL templates are not installed.")

        run_root = Path(
            os.environ.get(
                "PRIME_RL_LOCAL_RUN_DIR",
                f"/tmp/prime-rl/{cluster.cluster_id}/rank-{cluster.rank}",
            )
        )
        config_dir = run_root / "configs"
        write_subconfigs(config, config_dir)
        _ensure_run_directories(config)

        runtime_venv = _runtime_venv(project_dir)
        context = regular_multi_node_template_context(config, config_dir)
        context.update(
            {
                "project_dir": shlex.quote(project_dir.as_posix()),
                "runtime_venv": shlex.quote(runtime_venv.as_posix()),
                "config_dir": shlex.quote(config_dir.as_posix()),
                "output_dir": shlex.quote(config.output_dir.as_posix()),
                "orchestrator_output_dir": shlex.quote(config.orchestrator.output_dir.as_posix()),
                "node_addresses": shlex.quote(" ".join(cluster.addresses)),
                "node_rank": cluster.rank,
                "local_address": shlex.quote(cluster.local_address),
                "cluster_id": shlex.quote(cluster.cluster_id),
                "pool_namespace": shlex.quote(cluster.pool_namespace),
                "cleanup_grace_period": 0,
            }
        )

        environment = Environment(
            loader=FileSystemLoader(templates_dir),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        script_path = run_root / "cluster_node.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(environment.get_template("allocated_cluster_node.sh.j2").render(**context))
        if config.dry_run:
            return 0

        return _execute_script(script_path)


def main() -> None:
    set_proc_title("ClusterNode")
    if len(sys.argv) < 2 or sys.argv[1] not in {"prepare", "run"}:
        raise ValueError("Usage: python -m prime_rl.entrypoints.cluster_node {prepare|run} @ CONFIG")

    action = sys.argv.pop(1)
    config = cli(RLConfig)
    cluster = AllocatedCluster.from_environment()
    if action == "prepare":
        token = prepare(config, cluster)
        sys.stdout.write(f"{_PREPARATION_TOKEN_PREFIX}{token}\n")
        return
    raise SystemExit(run_node(config, cluster))


def _validate_config(config: RLConfig, cluster: AllocatedCluster) -> None:
    if config.deployment.type != "multi_node":
        raise ValueError("The allocated-cluster entrypoint requires deployment.type = 'multi_node'.")
    if config.slurm is not None:
        raise ValueError("The allocated-cluster entrypoint does not accept a Slurm configuration.")
    if config.inference is None or config.inference.deployment.type != "single_node":
        raise ValueError("Allocated clusters currently require regular single-node inference.")
    expected_nodes = config.deployment.num_train_nodes + config.deployment.total_infer_nodes
    if expected_nodes != cluster.node_count:
        raise ValueError(
            f"PrimeRL deployment requires {expected_nodes} nodes, but allocator supplied {cluster.node_count}."
        )


def _require_preparation_token(
    config: RLConfig,
    cluster: AllocatedCluster,
) -> None:
    expected = _preparation_token(config, cluster)
    actual = os.environ.get("PRIME_RL_PREPARATION_TOKEN")
    if actual is None:
        raise RuntimeError(
            "Allocated-cluster preparation token is missing; run prepare and forward its token to every worker."
        )
    if actual != expected:
        raise RuntimeError("Allocated-cluster configuration or topology changed after preparation.")


def _preparation_token(
    config: RLConfig,
    cluster: AllocatedCluster,
) -> str:
    config_json = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = json.dumps(
        {
            "schema_version": 1,
            "cluster_id": cluster.cluster_id,
            "node_addresses": list(cluster.addresses),
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _project_dir() -> Path:
    raw_path = os.environ.get("PRIME_RL_PROJECT_DIR")
    if not raw_path:
        raise ValueError("PRIME_RL_PROJECT_DIR must identify the PrimeRL checkout.")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError("PRIME_RL_PROJECT_DIR must be absolute.")
    if not (path / "pyproject.toml").is_file():
        raise ValueError(f"PRIME_RL_PROJECT_DIR is not a PrimeRL checkout: {path}")
    return path.resolve()


def _runtime_venv(project_dir: Path) -> Path:
    path = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", project_dir / ".venv"))
    if not path.is_absolute():
        raise ValueError("UV_PROJECT_ENVIRONMENT must be absolute when set.")
    if not (path / "bin/activate").is_file():
        raise ValueError(f"PrimeRL virtual environment does not exist: {path}")
    return path.resolve()


def _ensure_run_directories(config: RLConfig) -> None:
    (config.output_dir / "logs" / "trainer").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "logs" / "inference").mkdir(parents=True, exist_ok=True)


def _execute_script(script_path: Path) -> int:
    return subprocess.run(["bash", script_path.as_posix()], check=False).returncode


if __name__ == "__main__":
    main()
