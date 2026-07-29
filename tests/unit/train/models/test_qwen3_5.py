import inspect
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer import model as model_loading
from prime_rl.trainer.models.layers.attn import FlashAttention, substitute_ring_attn
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedFlashAttention
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.cp import setup_model_cp


def _tiny_text_config(attn_impl: str = "flash_attention_2") -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
    )
    config._attn_implementation = attn_impl
    return config


def _tiny_vlm_config(attn_impl: str = "flash_attention_2") -> Qwen3_5Config:
    text_config = _tiny_text_config(attn_impl)
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=text_config.hidden_size,
    )
    config = Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = attn_impl
    config.text_config._attn_implementation = attn_impl
    return config


def _tiny_moe_config(attn_impl: str = "flash_attention_2") -> Qwen3_5MoeConfig:
    config = Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        use_grouped_mm=False,
    )
    config._attn_implementation = attn_impl
    return config


@pytest.mark.gpu
def test_qwen3_5_dense_matches_hf_state_keys_on_meta():
    config = _tiny_text_config()
    with torch.device("meta"):
        hf_model = HFQwen3_5ForCausalLM(config)
        prime_model = Qwen3_5ForCausalLM(config)

    assert set(prime_model.state_dict()) == set(hf_model.state_dict())
    for name, tensor in prime_model.state_dict().items():
        assert tensor.shape == hf_model.state_dict()[name].shape, name


@pytest.mark.parametrize("ambiguous_snapshot", [False, True])
def test_qwen3_5_dense_load_does_not_convert_identity_format(monkeypatch, tmp_path, ambiguous_snapshot):
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(_tiny_text_config())

    if ambiguous_snapshot:
        monkeypatch.setattr(type(model), "is_prime_state_dict", classmethod(lambda cls, state_dict: True))

    snapshot_keys = list(model.state_dict())
    load_state_dict = MagicMock()
    save_state_dict = MagicMock(side_effect=OSError("read-only model cache"))
    dcp_load = MagicMock()
    reader = object()

    monkeypatch.setattr(torch.distributed, "barrier", MagicMock())
    monkeypatch.setattr(model_loading, "load_state_dict_keys", lambda path: snapshot_keys)
    monkeypatch.setattr(model_loading, "load_state_dict", load_state_dict)
    monkeypatch.setattr(model_loading, "save_state_dict", save_state_dict)
    monkeypatch.setattr(model_loading, "HuggingFaceStorageReader", MagicMock(return_value=reader))
    monkeypatch.setattr(model_loading, "dcp_load", dcp_load)
    monkeypatch.setattr(model_loading, "_move_buffers_to_cuda", MagicMock())
    monkeypatch.setattr(model, "init_buffers_post_meta", MagicMock())

    model_loading.load_dcp_from_hf(
        model,
        ModelConfig(
            name=str(tmp_path),
            fsdp_cpu_offload=True,
            optim_cpu_offload=False,
        ),
        ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1),
    )

    load_state_dict.assert_not_called()
    save_state_dict.assert_not_called()
    assert dcp_load.call_args.kwargs["storage_reader"] is reader


def test_local_qwen3_6_dense_config_applies_qwen3_5_patches(monkeypatch, tmp_path):
    local_snapshot = tmp_path / "Qwen3.6-27B" / "snapshot"
    _tiny_vlm_config().save_pretrained(local_snapshot)

    class StubModel:
        @classmethod
        def from_config(cls, loaded_config, **kwargs):
            assert loaded_config.model_type == "qwen3_5"
            model = torch.nn.Module()
            model.lm_head = torch.nn.Linear(
                1,
                1,
                bias=False,
                device="meta",
                dtype=kwargs["dtype"],
            )
            return model

    patches = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    monkeypatch.setattr(model_loading, "_patch_qwen3_5_text_position_ids", patches[0])
    monkeypatch.setattr(model_loading, "_patch_qwen3_5_moe_conversion_mapping", patches[1])
    monkeypatch.setattr(model_loading, "_patch_qwen3_5_linear_attn_varlen", patches[2])
    monkeypatch.setattr(model_loading, "get_custom_vlm_cls", lambda _: StubModel)

    model_loading.get_model(
        ModelConfig(name=str(local_snapshot), attn="flash_attention_2"),
        device=torch.device("meta"),
    )

    for patch in patches:
        patch.assert_called_once_with()


@pytest.mark.parametrize("attn_impl", ["flash_attention_3", "kernels-community/vllm-flash-attn3"])
def test_qwen3_5_full_attention_uses_custom_class(attn_impl: str):
    config = _tiny_text_config(attn_impl=attn_impl)
    with torch.device("meta"):
        model = Qwen3_5Model(config)

    assert isinstance(model.layers[1].self_attn, Qwen3_5GatedFlashAttention)
    assert model.config._attn_implementation == "flash_attention_3"
    assert "ALL_ATTENTION_FUNCTIONS" not in inspect.getsource(type(model.layers[1].self_attn).forward)


def test_qwen3_5_moe_full_attention_normalizes_fa3_hub_alias():
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedFlashAttention,
        Qwen3_5MoeModel,
    )

    config = _tiny_moe_config(attn_impl="kernels-community/vllm-flash-attn3")
    with torch.device("meta"):
        model = Qwen3_5MoeModel(config)

    assert isinstance(model.layers[1].self_attn, Qwen3_5MoeGatedFlashAttention)
    assert model.config._attn_implementation == "flash_attention_3"


def test_qwen3_5_context_parallel_setup_chain_text_and_vlm():
    cp_group = MagicMock()

    text_model = Qwen3_5ForCausalLM(_tiny_text_config())
    linear_layer = text_model.model.layers[0]
    text_model.model.layers[0] = torch.nn.Sequential(linear_layer)
    setup_model_cp(text_model, cp_group, cp_rank=1, cp_world_size=2)
    assert text_model.model._cp_group is cp_group
    assert text_model.model._cp_rank == 1
    assert text_model.model._cp_world_size == 2
    assert linear_layer.linear_attn.cp_group is cp_group

    vlm_config = _tiny_vlm_config()
    vlm_config.vision_config._attn_implementation = "sdpa"
    vlm_config.vision_config._attn_implementation_internal = "sdpa"
    with torch.device("meta"):
        vlm_model = Qwen3_5ForCausalLM(vlm_config)
    setup_model_cp(vlm_model, cp_group, cp_rank=0, cp_world_size=2)
    assert vlm_model.model.language_model._cp_group is cp_group
    assert vlm_model.model.language_model.layers[0].linear_attn.cp_world_size == 2


def test_setup_model_cp_requires_hook_only_for_hybrid_models():
    class HybridLayer(torch.nn.Module):
        layer_type = "linear_attention"

    class Inner:
        layers = torch.nn.Sequential(torch.nn.Sequential(HybridLayer()))

    class HybridNoHookModel:
        model = Inner()

    with pytest.raises(ValueError, match="set_context_parallel_attributes"):
        setup_model_cp(HybridNoHookModel(), MagicMock(), cp_rank=0, cp_world_size=2)

    class SoftmaxOnlyModel:
        pass

    setup_model_cp(SoftmaxOnlyModel(), MagicMock(), cp_rank=0, cp_world_size=2)


def test_qwen3_5_ring_patches_dense_flash_attention():
    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention
    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedFlashAttention

    originals = {
        cls: cls._compute_attention
        for cls in (FlashAttention, AfmoeFlashAttention, Qwen3_5MoeGatedFlashAttention, Qwen3_5GatedFlashAttention)
    }
    try:
        substitute_ring_attn(process_group=MagicMock(), heads_k_stride=1)
        assert Qwen3_5GatedFlashAttention._compute_attention is not originals[Qwen3_5GatedFlashAttention]
    finally:
        for cls, method in originals.items():
            cls._compute_attention = method
