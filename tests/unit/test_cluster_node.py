import json
import subprocess
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints import cluster_node
from prime_rl.entrypoints.cluster_node import AllocatedCluster
from prime_rl.entrypoints.inference import (
    write_slurm_script as write_inference_slurm_script,
)
from prime_rl.entrypoints.rl import rl, write_slurm_script, write_subconfigs


def _config(
    tmp_path: Path,
    *,
    slurm: dict | None = None,
    env_vars: dict[str, str] | None = None,
) -> RLConfig:
    data = {
        "trainer": {},
        "orchestrator": {},
        "inference": {"parallel": {"tp": 8}},
        "deployment": {
            "type": "multi_node",
            "gpus_per_node": 8,
            "num_train_nodes": 1,
            "num_infer_nodes": 1,
        },
        "output_dir": tmp_path / "output",
    }
    if slurm is not None:
        data["slurm"] = slurm
    if env_vars is not None:
        data["env_vars"] = env_vars
    return RLConfig.model_validate(data)


def _cluster(**overrides) -> AllocatedCluster:
    values = {
        "rank": 0,
        "node_count": 2,
        "addresses": ("10.0.0.1", "10.0.0.2"),
        "local_address": "10.0.0.1",
        "cluster_id": "cluster-123",
        "pool_namespace": "allocated",
    }
    values.update(overrides)
    return AllocatedCluster(**values)


def test_multi_node_config_does_not_require_slurm(tmp_path):
    config = _config(tmp_path)

    assert config.deployment.type == "multi_node"
    assert config.slurm is None


def test_config_rejects_allocator_owned_environment(tmp_path):
    with pytest.raises(ValueError, match="launcher-managed vars"):
        _config(tmp_path, env_vars={"PRIME_RL_NODE_RANK": "0"})


def test_rl_rejects_external_multi_node_before_preparation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    prepared = False

    def mark_prepared(_config):
        nonlocal prepared
        prepared = True

    monkeypatch.setattr("prime_rl.entrypoints.rl.prepare_rl_run", mark_prepared)

    with pytest.raises(ValueError, match="allocated-cluster entrypoint"):
        rl(config)

    assert not prepared


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"node_count": 3}, "Expected 3 node addresses"),
        ({"addresses": ("10.0.0.1", "10.0.0.1")}, "must be unique"),
        ({"rank": 2}, "outside world size"),
        ({"local_address": "10.0.0.2"}, "must equal"),
        (
            {
                "addresses": ("[fd00::1]", "10.0.0.2"),
                "local_address": "[fd00::1]",
            },
            "unsupported character",
        ),
        ({"cluster_id": "../unsafe"}, "safe nonempty path token"),
    ],
)
def test_allocated_cluster_rejects_invalid_topology(overrides, message):
    with pytest.raises(ValueError, match=message):
        _cluster(**overrides).validate()


def test_allocated_cluster_reads_explicit_environment(monkeypatch):
    values = {
        "PRIME_RL_NODE_RANK": "1",
        "PRIME_RL_NODE_COUNT": "2",
        "PRIME_RL_NODE_ADDRESSES_JSON": json.dumps(["10.0.0.1", "10.0.0.2"]),
        "PRIME_RL_LOCAL_ADDRESS": "10.0.0.2",
        "PRIME_RL_CLUSTER_ID": "cluster-123",
        "PRIME_RL_POOL_NAMESPACE": "modal",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    result = AllocatedCluster.from_environment()

    assert result == _cluster(rank=1, local_address="10.0.0.2", pool_namespace="modal")


def test_prepare_is_idempotent_for_log_links(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _configure_external_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(cluster_node, "prepare_rl_run", lambda _config: None)

    cluster_node.prepare(config, _cluster())
    cluster_node.prepare(config, _cluster())

    assert (config.output_dir / "logs" / "trainer.log").readlink() == Path("trainer/node_0.log")
    assert (config.output_dir / "logs" / "inference.log").readlink() == Path("inference/node_0.log")
    assert (config.output_dir / "configs" / "trainer.toml").is_file()
    assert (config.output_dir / "configs" / "orchestrator.toml").is_file()
    assert (config.output_dir / "configs" / "inference.toml").is_file()


def test_prepare_rejects_nonzero_rank(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="must run on rank 0"):
        cluster_node.prepare(
            config,
            _cluster(rank=1, local_address="10.0.0.2"),
        )


def test_external_node_renders_shared_placement_worker(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _prepare_external_run(tmp_path, monkeypatch, config)
    monkeypatch.setenv("PRIME_RL_LOCAL_RUN_DIR", (tmp_path / "rank-0").as_posix())
    monkeypatch.setattr(cluster_node, "_execute_script", lambda _path: 0)

    assert cluster_node.run_node(config, _cluster()) == 0

    script = (tmp_path / "rank-0" / "cluster_node.sh").read_text()
    assert "HOSTNAMES_STR='10.0.0.1 10.0.0.2'" in script
    assert "--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT" in script
    assert "--rdzv-id=job_$PRIME_RL_CLUSTER_ID" in script
    assert "WANDB_SHARED_RUN_ID=${WANDB_SHARED_RUN_ID:-$PRIME_RL_CLUSTER_ID}" in script
    assert "SLURM_" not in script
    assert script.index("[ -f .env ] && source .env") < script.index("export PRIME_RL_NODE_RANK=0")
    assert script.index("[ -f .env ] && source .env") < script.index("export UV_PROJECT_ENVIRONMENT=")
    subprocess.run(
        ["bash", "-n", (tmp_path / "rank-0" / "cluster_node.sh").as_posix()],
        check=True,
    )


def test_external_node_requires_matching_preparation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _configure_external_runtime(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="preparation token is missing"):
        cluster_node.run_node(config, _cluster())

    monkeypatch.setattr(cluster_node, "prepare_rl_run", lambda _config: None)
    token = cluster_node.prepare(config, _cluster())
    monkeypatch.setenv("PRIME_RL_PREPARATION_TOKEN", token)
    changed_cluster = _cluster(
        addresses=("10.0.0.3", "10.0.0.4"),
        local_address="10.0.0.3",
    )
    with pytest.raises(RuntimeError, match="topology changed"):
        cluster_node.run_node(config, changed_cluster)


def test_external_dry_run_renders_without_execution(tmp_path, monkeypatch):
    config = _config(tmp_path).model_copy(update={"dry_run": True})
    _prepare_external_run(tmp_path, monkeypatch, config)
    monkeypatch.setenv("PRIME_RL_LOCAL_RUN_DIR", (tmp_path / "rank-0").as_posix())
    monkeypatch.setattr(
        cluster_node,
        "_execute_script",
        lambda _path: pytest.fail("dry-run executed the worker"),
    )

    assert cluster_node.run_node(config, _cluster()) == 0
    assert (tmp_path / "rank-0" / "cluster_node.sh").is_file()


def test_slurm_adapter_invokes_shared_worker(tmp_path):
    config = _config(tmp_path, slurm={})
    config_dir = tmp_path / "configs"
    write_subconfigs(config, config_dir)
    script_path = tmp_path / "rl.sbatch"

    write_slurm_script(config, config_dir, script_path)

    script = script_path.read_text()
    assert "srun --kill-on-bad-exit=1" in script
    assert "export PRIME_RL_NODE_RANK=$SLURM_PROCID" in script
    assert "--node-rank=$TRAIN_NODE_RANK" in script
    assert "--rdzv-id=job_$PRIME_RL_CLUSTER_ID" in script
    subprocess.run(["bash", "-n", script_path.as_posix()], check=True)


def test_custom_slurm_template_can_include_shared_worker(tmp_path):
    config = _config(tmp_path, slurm={})
    custom_template = tmp_path / "custom.sbatch.j2"
    custom_template.write_text(
        "infer_nodes_per_replica={{ infer_nodes_per_replica }}\n{% include '_multi_node_rank.sh.j2' %}"
    )
    config.slurm.template_path = custom_template
    config_dir = tmp_path / "configs"
    write_subconfigs(config, config_dir)
    script_path = tmp_path / "rl.sbatch"

    write_slurm_script(config, config_dir, script_path)

    script = script_path.read_text()
    assert "infer_nodes_per_replica=1" in script
    assert "--node-rank=$TRAIN_NODE_RANK" in script


def test_disaggregated_slurm_still_renders_shared_worker(tmp_path):
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {
                "parallel": {"tp": 8},
                "deployment": {
                    "type": "disaggregated",
                    "prefill_nodes_per_replica": 1,
                    "decode_nodes_per_replica": 1,
                },
            },
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 8,
                "num_train_nodes": 1,
            },
            "slurm": {},
            "output_dir": tmp_path / "output",
        }
    )
    config_dir = tmp_path / "configs"
    write_subconfigs(config, config_dir)
    script_path = tmp_path / "rl.sbatch"

    write_slurm_script(config, config_dir, script_path)

    script = script_path.read_text()
    assert 'ROLE="prefill"' in script
    assert 'ROLE="decode"' in script
    assert "srun --kill-on-bad-exit=1" in script
    assert "export PRIME_RL_POOL_NAMESPACE=slurm" in script
    subprocess.run(["bash", "-n", script_path.as_posix()], check=True)


def test_regular_llmd_slurm_preserves_endpoint_rendering(tmp_path):
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {
                "parallel": {"tp": 4},
                "router": {"type": "llm-d"},
                "deployment": {"type": "multi_node", "num_nodes": 1},
            },
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 8,
                "num_train_nodes": 1,
            },
            "slurm": {},
            "output_dir": tmp_path / "output",
        }
    )
    config_dir = tmp_path / "configs"
    write_subconfigs(config, config_dir)
    script_path = tmp_path / "rl.sbatch"

    write_slurm_script(config, config_dir, script_path)

    script = script_path.read_text()
    assert "__LLMD_ADDR_0__" in script
    assert "backend-0-rank-0" in script
    assert "backend-0-rank-1" in script
    assert "PRIME_RL_POOL_NAMESPACE=slurm" in script
    subprocess.run(["bash", "-n", script_path.as_posix()], check=True)


def test_inference_only_slurm_populates_neutral_partial_contract(tmp_path):
    config = InferenceConfig.model_validate(
        {
            "parallel": {"tp": 8},
            "deployment": {
                "type": "multi_node",
                "num_nodes": 2,
                "gpus_per_node": 8,
            },
            "kv_cache_offload": {
                "type": "mooncake",
                "cpu": {"num_bytes": 1024},
            },
            "slurm": {},
            "output_dir": tmp_path / "output",
        }
    )
    script_path = tmp_path / "inference.sbatch"

    write_inference_slurm_script(
        config,
        tmp_path / "inference.toml",
        script_path,
    )

    script = script_path.read_text()
    contract = script.index("export PRIME_RL_CLUSTER_ID=$SLURM_JOB_ID")
    mooncake = script.index('MC_HEAD="${HOSTNAMES[0]}"')
    assert contract < mooncake
    assert "export PRIME_RL_POOL_NAMESPACE=slurm" in script
    assert 'read -ra HOSTNAMES <<< "$HOSTNAMES_STR"' in script
    subprocess.run(["bash", "-n", script_path.as_posix()], check=True)


def _prepare_external_run(tmp_path, monkeypatch, config) -> None:
    _configure_external_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(cluster_node, "prepare_rl_run", lambda _config: None)
    token = cluster_node.prepare(config, _cluster())
    monkeypatch.setenv("PRIME_RL_PREPARATION_TOKEN", token)


def _configure_external_runtime(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / "prime-rl"
    project_dir.mkdir(exist_ok=True)
    (project_dir / "pyproject.toml").touch()
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
    (venv_dir / "bin/activate").touch()
    monkeypatch.setenv("PRIME_RL_PROJECT_DIR", project_dir.as_posix())
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", venv_dir.as_posix())
