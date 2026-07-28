import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import verifiers.v1 as vf
from huggingface_hub.constants import HF_HUB_CACHE
from renderers import Qwen35RendererConfig, Qwen36RendererConfig, create_renderer
from renderers.base import ParsedToolCall, ToolCallParseStatus
from transformers import AutoTokenizer
from verifiers.v1 import graph
from verifiers.v1.clients import TrainClient
from verifiers.v1.clients.train import tool_to_wire
from verifiers.v1.dialects import ChatDialect

from prime_rl.orchestrator.trajectories import trace_to_samples

_QWEN35_SNAPSHOT = (
    "Qwen/Qwen3.5-0.8B",
    "2fc06364715b967f1860aea9cf38778875588b17",
)
_QWEN36_SNAPSHOT = (
    "Qwen/Qwen3.6-27B",
    "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
)


def _cached_snapshot(repo_id: str, revision: str) -> Path:
    path = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        pytest.skip(
            f"requires immutable local snapshot {repo_id}@{revision}; "
            f"missing {', '.join(missing)} under {path}"
        )
    return path


def _token_digest(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _trace() -> vf.Trace:
    return vf.Trace(task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="compatibility")))


def _completion_through_stop(renderer, token_ids: list[int]) -> list[int]:
    stop_ids = set(renderer.get_stop_token_ids())
    stop = next((index for index, token_id in enumerate(token_ids) if token_id in stop_ids), None)
    return token_ids if stop is None else token_ids[: stop + 1]


def test_local_qwen3_6_dense_config_applies_qwen3_5_patches(monkeypatch, tmp_path):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer import model as model_loading

    local_snapshot = tmp_path / "Qwen3.6-27B" / "snapshot"
    config = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
        ),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config.save_pretrained(local_snapshot)

    class StubModel:
        @classmethod
        def from_config(cls, loaded_config, **kwargs):
            assert loaded_config.model_type == "qwen3_5"
            model = torch.nn.Module()
            model.lm_head = torch.nn.Linear(1, 1, bias=False, device="meta", dtype=kwargs["dtype"])
            return model

    patch_functions = [
        "_patch_qwen3_5_text_position_ids",
        "_patch_qwen3_5_moe_conversion_mapping",
        "_patch_qwen3_5_linear_attn_varlen",
    ]
    patch_mocks = [MagicMock() for _ in patch_functions]
    for name, mock in zip(patch_functions, patch_mocks, strict=True):
        monkeypatch.setattr(model_loading, name, mock)
    monkeypatch.setattr(model_loading, "get_custom_vlm_cls", lambda _: StubModel)

    model_loading.get_model(
        ModelConfig(name=str(local_snapshot), attn="flash_attention_2"),
        device=torch.device("meta"),
    )

    for mock in patch_mocks:
        mock.assert_called_once_with()


def test_qwen3_5_non_thinking_initial_render_is_pinned():
    snapshot = _cached_snapshot(*_QWEN35_SNAPSHOT)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    renderer = create_renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))
    messages = [{"role": "user", "content": "Reply with pong."}]

    rendered = renderer.render(messages, add_generation_prompt=True)

    assert renderer.config.enable_thinking is False
    assert rendered.token_ids == renderer.render(messages, add_generation_prompt=True).token_ids
    assert len(rendered.token_ids) == 16
    assert _token_digest(rendered.token_ids) == "ef2787a799cbccc7e65d3d0a00926c26592681fc82fcc91fbc821a73e6f661e3"
    assert tokenizer.decode(rendered.token_ids, skip_special_tokens=False).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_qwen3_6_thinking_tool_cycle_preserves_train_sample_alignment():
    snapshot = _cached_snapshot(*_QWEN36_SNAPSHOT)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    trace = _trace()
    user = vf.UserMessage(content="What is the weather in Paris?")
    tool = vf.Tool(
        name="get_weather",
        description="Get weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "metric": {"type": "boolean"},
            },
            "required": ["city", "metric"],
        },
    )
    body = {"tools": [tool_to_wire(tool)]}
    client = TrainClient(
        AsyncMock(),
        config=Qwen36RendererConfig(enable_thinking=True),
        renderer_model_name=str(snapshot),
    )
    prompt_ids_by_turn: list[list[int]] = []
    completion_ids_by_turn: list[list[int]] = []
    completion_logprobs_by_turn: list[list[float]] = []

    async def fake_generate(**kwargs):
        renderer = kwargs["renderer"]
        if kwargs["prompt_ids"] is None:
            attribution = renderer.render(
                kwargs["messages"],
                tools=kwargs["tools"],
                add_generation_prompt=True,
            )
            prompt_ids = attribution.token_ids
        else:
            attribution = kwargs["prompt_attribution"]
            prompt_ids = kwargs["prompt_ids"]
        assert attribution is not None
        prompt_ids_by_turn.append(prompt_ids)

        if not completion_ids_by_turn:
            assistant = {
                "role": "assistant",
                "reasoning_content": "I should use the weather tool.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris", "metric": True},
                        },
                    }
                ],
            }
            full_turn = renderer.render(
                [*kwargs["messages"], assistant],
                tools=kwargs["tools"],
                add_generation_prompt=False,
            )
            completion_ids = _completion_through_stop(renderer, full_turn.token_ids[len(prompt_ids) :])
            result = {
                "content": "",
                "reasoning_content": "I should use the weather tool.",
                "tool_calls": [
                    ParsedToolCall(
                        raw="",
                        name="get_weather",
                        arguments={"city": "Paris", "metric": True},
                        status=ToolCallParseStatus.OK,
                        id="call_weather",
                    )
                ],
                "finish_reason": "tool_calls",
            }
        else:
            completion_ids = tokenizer.encode(
                "The tool returned the answer.</think>\n\nIt is 20 C.",
                add_special_tokens=False,
            ) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
            result = {
                "content": "It is 20 C.",
                "reasoning_content": "The tool returned the answer.",
                "tool_calls": [],
                "finish_reason": "stop",
            }

        turn_number = len(completion_ids_by_turn) + 1
        completion_logprobs = [-turn_number - index / 100 for index in range(len(completion_ids))]
        completion_ids_by_turn.append(completion_ids)
        completion_logprobs_by_turn.append(completion_logprobs)
        return {
            "request_id": f"qwen36-{turn_number}",
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_logprobs": completion_logprobs,
            "prompt_attribution": attribution,
            **result,
        }

    async def run_tool_cycle():
        with patch("renderers.client.generate", side_effect=fake_generate):
            first_turn = graph.prepare_turn(trace, [user])
            first_response = await client.get_response(
                ChatDialect(),
                body,
                "runtime-model",
                vf.SamplingConfig(),
                turn=first_turn,
            )
            first_turn.commit(first_response, [tool])

            tool_response = vf.ToolMessage(
                tool_call_id="call_weather",
                name="get_weather",
                content="20 C",
            )
            second_turn = graph.prepare_turn(trace, [user, first_response.message, tool_response])
            second_response = await client.get_response(
                ChatDialect(),
                body,
                "runtime-model",
                vf.SamplingConfig(),
                turn=second_turn,
            )
            second_turn.commit(second_response, [tool])

    asyncio.run(run_tool_cycle())

    initial_prompt, bridged_prompt = prompt_ids_by_turn
    first_completion, second_completion = completion_ids_by_turn
    assert client._pool is not None
    with client._pool.checkout() as renderer:
        assert renderer.effective_thinking_retention == "tool_cycle"
    assert len(initial_prompt) == 286
    assert _token_digest(initial_prompt) == "45a917f11ca4404504698df9af475245c9f83a6d35b7865b40875701407cd5b9"
    assert len(bridged_prompt) == 352
    assert _token_digest(bridged_prompt) == "96709c4a5f062d98c4e89806a1e781d258cc9dec8d8b28a6442e381d1cac5e62"
    assert bridged_prompt[: len(initial_prompt) + len(first_completion)] == initial_prompt + first_completion

    [sample] = trace_to_samples(trace, env_name="qwen-compatibility")
    expected_logprobs = completion_logprobs_by_turn[0] + completion_logprobs_by_turn[1]
    assert sample.token_ids == bridged_prompt + second_completion
    assert len(sample.token_ids) == len(sample.mask) == len(sample.logprobs)
    assert sum(sample.mask) == len(first_completion) + len(second_completion)
    assert [value for value, sampled in zip(sample.logprobs, sample.mask, strict=True) if sampled] == expected_logprobs
