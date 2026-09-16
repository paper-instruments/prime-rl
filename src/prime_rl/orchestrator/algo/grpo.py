from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        else:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            output_denominator = (
                length_penalty.output_token_reference_length
                if length_penalty.output_token_reference_length is not None
                else output.max().clamp(min=1)
            )
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output_denominator)
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            if length_penalty.num_returned_tool_tokens_weight > 0:
                returned_tool_counts = []
                for rollout in group:
                    count = rollout.metrics.get("returned_tool_tokens")
                    if (
                        isinstance(count, bool)
                        or not isinstance(count, (int, float))
                        or not math.isfinite(count)
                        or count < 0
                        or count != int(count)
                    ):
                        raise ValueError(
                            "returned_tool_tokens must be a finite, non-negative integer-valued metric "
                            f"(env '{rollout.env_name}', got {count!r})"
                        )
                    returned_tool_counts.append(count)
                reference_length = length_penalty.returned_tool_token_reference_length
                assert reference_length is not None
                returned_tool_tokens = torch.tensor(returned_tool_counts, dtype=rewards.dtype)
                penalty_frac += length_penalty.num_returned_tool_tokens_weight * (returned_tool_tokens / reference_length)
            penalty = rewards.mean().clamp_min(0) * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
