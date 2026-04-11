# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .activations import swiglu
from .base import BaseModelArgs, scaled_dot_product_attention
from .cache import BlockKVCache, CacheList, KVCache
from .rope_utils import initialize_rope

DEFAULT_BLOCK_SIZE = 32
DEFAULT_SMALL_BLOCK_SIZE = 8
DEFAULT_THRESHOLD = 0.9
DEFAULT_MIN_UNMASKS_PER_STEP = 1


def make_block_attention_mask(
    seq_len: int,
    block_size: int,
    cache_seq_len: int = 0,
    left_padding: Optional[mx.array] = None,
    right_padding: Optional[mx.array] = None,
) -> Optional[mx.array]:
    if seq_len <= 0:
        return None

    q_idx = mx.arange(cache_seq_len, cache_seq_len + seq_len)
    kv_idx = mx.arange(cache_seq_len + seq_len)

    if left_padding is None:
        mask = (q_idx[:, None] // block_size) >= (kv_idx[None, :] // block_size)
    else:
        left_padding = mx.array(left_padding)
        q_blocks = (q_idx[None, None, :, None] - left_padding[:, None, None, None]) // (
            block_size
        )
        kv_blocks = (
            kv_idx[None, None, None, :] - left_padding[:, None, None, None]
        ) // block_size
        mask = q_blocks >= kv_blocks
        mask = mask & (kv_idx[None, None, None, :] >= left_padding[:, None, None, None])

    if right_padding is not None:
        right_padding = mx.array(right_padding)
        limit = cache_seq_len + seq_len - right_padding
        mask = mask & (kv_idx[None, None, None, :] < limit[:, None, None, None])

    return mask


def shift_diffusion_logits(logits: mx.array) -> mx.array:
    return mx.concatenate([logits[:, :1, :], logits[:, :-1, :]], axis=1)


def first_stop_offset(tokens: Sequence[int], stop_ids: set[int]) -> Optional[int]:
    for i, token in enumerate(tokens):
        if token in stop_ids:
            return i
    return None


def _split_cache_entry(cache_entry):
    if cache_entry is None:
        return None, None
    if isinstance(cache_entry, CacheList):
        return cache_entry[0], cache_entry[1]
    return cache_entry, None


def _cache_rope_offset(cache) -> Union[int, mx.array]:
    if cache is None:
        return 0
    return cache.offset


def _cache_mask_offset(cache) -> int:
    if cache is None:
        return 0
    return cache.size()


def _cache_left_padding(cache) -> Optional[mx.array]:
    return getattr(cache, "left_padding", None)


def _cache_right_padding(cache) -> Optional[mx.array]:
    return getattr(cache, "_right_padding", None)


def _cache_kv(cache):
    if cache is None or cache.empty():
        return None, None
    state = cache.state
    return state[0], state[1]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    mask_token_id: int = 151665
    bd_size: int = 32


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        replace_position: Optional[int] = None,
        update_past_key_values: bool = True,
        use_block_cache: bool = False,
    ) -> mx.array:
        B, L, _ = x.shape

        history_cache, block_cache = _split_cache_entry(cache)
        if not use_block_cache:
            block_cache = None

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        history_offset = _cache_rope_offset(history_cache)
        position_offset = history_offset + (replace_position or 0)

        if history_cache is not None or block_cache is not None:
            queries = self.rope(queries, offset=position_offset)
            keys = self.rope(keys, offset=position_offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if block_cache is not None:
            if block_cache.empty() or replace_position is None:
                keys, values = block_cache.update_and_fetch(keys, values)
            else:
                keys, values = block_cache.update_slice(keys, values, replace_position)

            past_keys, past_values = _cache_kv(history_cache)
            if past_keys is not None:
                keys = mx.concatenate([past_keys, keys], axis=-2)
                values = mx.concatenate([past_values, values], axis=-2)
        elif history_cache is not None:
            if update_past_key_values:
                keys, values = history_cache.update_and_fetch(keys, values)
            else:
                past_keys, past_values = _cache_kv(history_cache)
                if past_keys is not None:
                    keys = mx.concatenate([past_keys, keys], axis=-2)
                    values = mx.concatenate([past_values, values], axis=-2)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        replace_position: Optional[int] = None,
        update_past_key_values: bool = True,
        use_block_cache: bool = False,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            replace_position=replace_position,
            update_past_key_values=update_past_key_values,
            use_block_cache=use_block_cache,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class FastDLLMQwenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Optional[Any]]] = None,
        input_embeddings: Optional[mx.array] = None,
        block_size: Optional[int] = None,
        replace_position: Optional[int] = None,
        update_past_key_values: bool = True,
        use_block_cache: bool = False,
    ) -> mx.array:
        block_size = block_size or self.args.bd_size
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        history_cache, block_cache = _split_cache_entry(cache[0]) if cache else (None, None)
        cache_seq_len = _cache_mask_offset(history_cache)
        left_padding = _cache_left_padding(history_cache)
        right_padding = _cache_right_padding(history_cache)

        if not use_block_cache:
            for layer_cache in cache:
                _, layer_block_cache = _split_cache_entry(layer_cache)
                if layer_block_cache is not None and not layer_block_cache.empty():
                    layer_block_cache.clear()

        has_active_block_cache = (
            use_block_cache and block_cache is not None and not block_cache.empty()
        )
        mask = None
        if not has_active_block_cache:
            mask = make_block_attention_mask(
                h.shape[1],
                block_size,
                cache_seq_len=cache_seq_len,
                left_padding=left_padding,
                right_padding=right_padding,
            )

        for layer, layer_cache in zip(self.layers, cache):
            h = layer(
                h,
                mask=mask,
                cache=layer_cache,
                replace_position=replace_position,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache,
            )

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FastDLLMQwenModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Optional[Any]]] = None,
        input_embeddings: Optional[mx.array] = None,
        block_size: Optional[int] = None,
        replace_position: Optional[int] = None,
        update_past_key_values: bool = True,
        use_block_cache: bool = False,
    ) -> mx.array:
        out = self.model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            block_size=block_size,
            replace_position=replace_position,
            update_past_key_values=update_past_key_values,
            use_block_cache=use_block_cache,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [CacheList(KVCache(), BlockKVCache()) for _ in self.model.layers]

    def clear_block_caches(self, cache: Optional[List[Optional[Any]]]):
        if cache is None:
            return
        for layer_cache in cache:
            _, block_cache = _split_cache_entry(layer_cache)
            if block_cache is not None:
                block_cache.clear()

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        n_devices = group.size()
        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= n_devices
            layer.self_attn.n_kv_heads //= n_devices

            layer.mlp.gate_proj = shard_linear(
                layer.mlp.gate_proj, "all-to-sharded", group=group
            )
            layer.mlp.down_proj = shard_linear(
                layer.mlp.down_proj, "sharded-to-all", group=group
            )
            layer.mlp.up_proj = shard_linear(
                layer.mlp.up_proj, "all-to-sharded", group=group
            )

    def _argmax_sampler(self, logprobs: mx.array) -> mx.array:
        return mx.argmax(logprobs, axis=-1)

    def _sample_positions(
        self, logits: mx.array, sampler
    ) -> tuple[mx.array, mx.array, mx.array]:
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tokens = sampler(logprobs).astype(mx.uint32)
        probs = mx.exp(mx.take_along_axis(logprobs, tokens[:, None], axis=-1).squeeze(-1))
        return tokens, probs, logprobs

    def diffusion_decode(
        self,
        prompt: mx.array,
        *,
        max_tokens: int = 256,
        sampler=None,
        prompt_progress_callback=None,
        eos_token_ids: Optional[Sequence[int]] = None,
        block_size: int = 32,
        small_block_size: int = 8,
        threshold: float = 1.0,
        mask_id: Optional[int] = None,
        use_block_cache: bool = False,
        min_unmasks_per_step: int = 1,
        **kwargs,
    ) -> Generator[tuple[int, mx.array], None, None]:
        if max_tokens < 0:
            raise ValueError("Fast_dLLM generation requires a finite max_tokens value.")
        if prompt.ndim != 1:
            raise ValueError("Fast_dLLM generation currently expects a single prompt.")
        if small_block_size <= 0 or block_size <= 0:
            raise ValueError("block_size and small_block_size must be positive.")
        if block_size % small_block_size != 0:
            raise ValueError("block_size must be divisible by small_block_size.")
        if min_unmasks_per_step <= 0:
            raise ValueError("min_unmasks_per_step must be positive.")

        sampler = sampler or self._argmax_sampler
        prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
        stop_ids = set(eos_token_ids or [])
        mask_id = self.args.mask_token_id if mask_id is None else mask_id
        if not (0 <= mask_id < self.args.vocab_size):
            raise ValueError(
                f"mask_id ({mask_id}) must be within the vocabulary size ({self.args.vocab_size})."
            )

        prompt = prompt.astype(mx.uint32)
        original_prompt_len = int(prompt.shape[0])
        if max_tokens == 0:
            return

        target_length = original_prompt_len + max_tokens
        prompt_progress_callback(0, original_prompt_len)

        input_ids = prompt[None]
        token_logprobs: Dict[int, mx.array] = {}
        past_key_values = self.make_cache()

        if input_ids.shape[1] > block_size:
            full_prefix_len = (input_ids.shape[1] // block_size) * block_size
            prefix = input_ids[:, :full_prefix_len]
            logits = self(
                prefix,
                cache=past_key_values,
                block_size=block_size,
                update_past_key_values=True,
            )
            if input_ids.shape[1] % block_size == 0:
                last_logits = logits[:, -1, :]
                last_logprobs = last_logits - mx.logsumexp(
                    last_logits, axis=-1, keepdims=True
                )
                next_token = sampler(last_logprobs).astype(mx.uint32)
                token_logprobs[input_ids.shape[1]] = last_logprobs.squeeze(0)
                input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        prompt_progress_callback(original_prompt_len, original_prompt_len)

        num_small_blocks = block_size // small_block_size
        while input_ids.shape[1] < target_length and (
            not stop_ids
            or first_stop_offset(input_ids[0, original_prompt_len:].tolist(), stop_ids)
            is None
        ):
            prompt_length = input_ids.shape[1]
            fill = block_size - (prompt_length % block_size)
            if fill == 0:
                fill = block_size

            x_init = mx.full((1, fill), mask_id, dtype=mx.uint32)
            x_t = mx.concatenate([input_ids, x_init], axis=1)

            while True:
                stop_offset = first_stop_offset(
                    x_t[0, original_prompt_len:].tolist(), stop_ids
                )
                if stop_offset is not None:
                    before_stop = x_t[
                        0, prompt_length : original_prompt_len + stop_offset
                    ].tolist()
                    if mask_id not in before_stop:
                        break

                mask_idx = x_t[:, -block_size:] == mask_id
                if int(mask_idx.sum().item()) == 0:
                    commit_block = x_t[:, -block_size:]
                    mx.eval(commit_block)
                    logits = self(
                        commit_block,
                        cache=past_key_values,
                        block_size=block_size,
                        update_past_key_values=True,
                    )
                    next_logits = logits[:, -1, :]
                    next_logprobs = next_logits - mx.logsumexp(
                        next_logits, axis=-1, keepdims=True
                    )
                    next_token = sampler(next_logprobs).astype(mx.uint32)
                    token_logprobs[x_t.shape[1]] = next_logprobs.squeeze(0)
                    x_t = mx.concatenate([x_t, next_token[:, None]], axis=1)
                    break

                for small_block_idx in range(num_small_blocks):
                    small_start = small_block_idx * small_block_size
                    small_end = small_start + small_block_size
                    start = small_start
                    end = small_end

                    while True:
                        block_mask_idx = x_t[:, -block_size:] == mask_id
                        segment_mask = block_mask_idx[:, start:end]
                        n_masked = int(segment_mask.sum().item())
                        if n_masked == 0:
                            break

                        stop_offset = first_stop_offset(
                            x_t[0, original_prompt_len:].tolist(), stop_ids
                        )
                        if stop_offset is not None:
                            before_stop = x_t[
                                0, prompt_length : original_prompt_len + stop_offset
                            ].tolist()
                            if mask_id not in before_stop:
                                break

                        if use_block_cache:
                            _, first_block_cache = _split_cache_entry(
                                past_key_values[0]
                            )
                            rebuild_block_cache = (
                                first_block_cache is None
                                or first_block_cache.empty()
                                or int(x_t[0, -block_size + small_start].item())
                                == mask_id
                            )
                            if rebuild_block_cache:
                                self.clear_block_caches(past_key_values)
                                logits = self(
                                    x_t[:, -block_size:],
                                    cache=past_key_values,
                                    block_size=block_size,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                )
                                logits = shift_diffusion_logits(logits)[:, start:end, :]
                            else:
                                slice_end = None if end == block_size else -block_size + end
                                logits = self(
                                    x_t[:, -block_size + start : slice_end],
                                    cache=past_key_values,
                                    block_size=block_size,
                                    replace_position=small_start,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                )
                                logits = shift_diffusion_logits(logits)
                        else:
                            input_block = x_t[:, -block_size:]
                            mx.eval(input_block)
                            logits = self(
                                input_block,
                                cache=past_key_values,
                                block_size=block_size,
                                update_past_key_values=False,
                            )
                            logits = shift_diffusion_logits(logits)[:, start:end, :]

                        # Evaluate intermediate results to keep the lazy graph
                        # small. Without these evals, MLX accumulates a large
                        # graph that is significantly slower to execute on some
                        # builds. See: https://github.com/ml-explore/mlx/issues/XXXX
                        mx.eval(logits)
                        sampled, probs, logprobs = self._sample_positions(
                            logits.reshape(-1, logits.shape[-1]), sampler
                        )
                        sampled = sampled.reshape(1, -1)
                        probs = probs.reshape(1, -1)
                        mx.eval(sampled, probs)

                        sampled_probs = mx.where(segment_mask, probs, -mx.inf)
                        unmask_idx = sampled_probs > threshold
                        force_unmasks = min(
                            min_unmasks_per_step, n_masked
                        )
                        if force_unmasks >= n_masked:
                            # All masked positions will be unmasked — skip top-k.
                            mx.eval(sampled)
                            unmask_idx = segment_mask
                        else:
                            # Materialize sampled_probs to avoid fusing argsort/
                            # argmax into a large lazy graph (crashes on some
                            # MLX builds).
                            mx.eval(sampled_probs)
                            if force_unmasks == 1:
                                top_idx = mx.argmax(sampled_probs, axis=-1)
                                unmask_idx[0, top_idx.item()] = True
                            else:
                                sp_copy = mx.array(sampled_probs)
                                for _ in range(force_unmasks):
                                    idx = mx.argmax(sp_copy, axis=-1).item()
                                    unmask_idx[0, idx] = True
                                    sp_copy[0, idx] = -mx.inf
                                    mx.eval(sp_copy)
                            unmask_idx = mx.logical_and(unmask_idx, segment_mask)

                        segment = x_t[:, -block_size + start : -block_size + end]
                        if end == block_size:
                            segment = x_t[:, -block_size + start :]
                        new_segment = mx.where(unmask_idx, sampled, segment)
                        mx.eval(new_segment)
                        if end == block_size:
                            x_t = mx.concatenate(
                                [x_t[:, : -block_size + start], new_segment], axis=1
                            )
                        else:
                            x_t = mx.concatenate(
                                [
                                    x_t[:, : -block_size + start],
                                    new_segment,
                                    x_t[:, -block_size + end :],
                                ],
                                axis=1,
                            )
                        mx.eval(x_t)

                        local_positions = [
                            i for i, flag in enumerate(unmask_idx[0].tolist()) if flag
                        ]
                        absolute_start = x_t.shape[1] - block_size + start
                        for pos in local_positions:
                            token_logprobs[absolute_start + pos] = logprobs[pos]

                    stop_offset = first_stop_offset(
                        x_t[0, original_prompt_len:].tolist(), stop_ids
                    )
                    if stop_offset is not None:
                        before_stop = x_t[
                            0, prompt_length : original_prompt_len + stop_offset
                        ].tolist()
                        if mask_id not in before_stop:
                            break

                if x_t.shape[1] >= target_length + block_size:
                    break

            input_ids = x_t

        generated = input_ids[0, original_prompt_len:].tolist()
        stop_offset = first_stop_offset(generated, stop_ids)
        if stop_offset is not None:
            generated = generated[: stop_offset + 1]
        generated = generated[:max_tokens]

        for i, token in enumerate(generated):
            absolute_pos = original_prompt_len + i
            yield token, token_logprobs[absolute_pos]
