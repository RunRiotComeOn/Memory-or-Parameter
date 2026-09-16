"""VERL dataset adapter for complete Qwen3.5 browser episodes.

Qwen3.5's chat template validates the complete role sequence, so VERL's
per-message ``MultiTurnSFTDataset`` tokenization cannot be used.  This adapter
tokenizes each complete multi-turn conversation and learns every assistant
action while masking all system and environment/user turns.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.chat_template import apply_chat_template


def _find_subsequence(sequence: list[int], needle: list[int], start: int = 0) -> int:
    for index in range(start, len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    return -1


def assistant_action_loss_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Mask action JSON inside every assistant turn in a rendered chat."""
    ids = input_ids.tolist()
    header = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    thinking_stub = tokenizer.encode(
        "<think>\n\n</think>\n\n", add_special_tokens=False
    )
    end_token = tokenizer.eos_token_id
    if end_token is None:
        raise ValueError("tokenizer has no assistant end token")

    mask = torch.zeros_like(input_ids)
    cursor = 0
    turns = 0
    while True:
        header_start = _find_subsequence(ids, header, cursor)
        if header_start < 0:
            break
        content_start = header_start + len(header)
        if ids[content_start : content_start + len(thinking_stub)] == thinking_stub:
            content_start += len(thinking_stub)
        try:
            content_end = ids.index(end_token, content_start)
        except ValueError as exc:
            raise ValueError("assistant turn has no end token") from exc
        if content_start >= content_end:
            raise ValueError("assistant action is empty")
        mask[content_start : content_end + 1] = 1
        turns += 1
        cursor = content_end + 1
    if turns == 0:
        raise ValueError("episode contains no assistant turns")
    return mask


class Qwen35ActionSFTDataset(MultiTurnSFTDataset):
    """Tokenize complete episodes with loss on every assistant action."""

    def _validate_messages(self, messages):
        expected_roles = ["system"] + [
            role
            for _ in range((len(messages) - 1) // 2)
            for role in ("user", "assistant")
        ]
        if len(messages) < 3 or len(messages) % 2 == 0 or [
            message["role"] for message in messages
        ] != expected_roles:
            raise ValueError(
                "Qwen35ActionSFTDataset expects system/(user/assistant)+ messages"
            )

    def _tokenize(self, messages, tools, enable_thinking, *, add_generation_prompt):
        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = dict(self.apply_chat_template_kwargs)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        encoded = apply_chat_template(
            processor,
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        return {key: value[0] for key, value in dict(encoded).items()}

    def _tools_for_item(self, item):
        return self.tools[item] if self.tools is not None else None

    def __getitem__(self, item):
        row = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row)
        self._validate_messages(messages)

        tools = self._tools_for_item(item)
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        complete = self._tokenize(messages, tools, enable_thinking, add_generation_prompt=False)
        input_ids = complete.pop("input_ids")
        attention_mask = complete.pop("attention_mask")

        loss_mask = assistant_action_loss_mask(input_ids, self.tokenizer)
        if not torch.any(loss_mask):
            raise ValueError("episode produced an empty assistant loss mask")

        # The processor is loaded for Qwen3.5 even for text-only examples.  Match
        # VERL's native multimodal position-id layout so the model sees the same
        # four RoPE axes it receives during inference.
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=None,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={sequence_length} is larger than max_length={self.max_length}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                position_ids = position_ids[..., -self.max_length :]
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]

        if self.pad_mode == DatasetPadMode.RIGHT and len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            loss_mask = F.pad(loss_mask, (0, pad_length), value=0)
            position_ids = F.pad(position_ids, (0, pad_length), value=0)

        result = {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
        if self.pad_mode == DatasetPadMode.RIGHT:
            result["attention_mask"] = attention_mask
        return result


class Qwen35TauSFTDataset(Qwen35ActionSFTDataset):
    """Complete tau-bench conversations with assistant/tool interleaving."""

    def _build_messages(self, example):
        raw = example[self.messages_key]
        if isinstance(raw, str):
            example = dict(example)
            example[self.messages_key] = json.loads(raw)
        return super()._build_messages(example)

    def _tools_for_item(self, item):
        tools = super()._tools_for_item(item)
        return json.loads(tools) if isinstance(tools, str) else tools

    def _validate_messages(self, messages):
        if len(messages) < 3 or messages[0].get("role") != "system":
            raise ValueError("tau SFT examples must begin with a system message")
        roles = {"system", "user", "assistant", "tool"}
        if any(message.get("role") not in roles for message in messages):
            raise ValueError("tau SFT example contains an unsupported role")
        if any(
            message.get("role") == "system" for message in messages[1:]
        ):
            raise ValueError("tau SFT example contains a non-initial system message")
        if not any(message.get("role") == "assistant" for message in messages[1:]):
            raise ValueError("tau SFT example contains no assistant target")
