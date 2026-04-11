# Copyright © 2025 Apple Inc.

import argparse
import time

import mlx.core as mx

from mlx_lm import batch_generate, load, stream_generate
from mlx_lm.generate import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MIN_UNMASKS_PER_STEP,
    DEFAULT_MODEL,
    DEFAULT_SMALL_BLOCK_SIZE,
    DEFAULT_THRESHOLD,
)
from mlx_lm.utils import pipeline_load, sharded_load


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM benchmarking script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--prompt-tokens",
        "-p",
        default=512,
        help="Length of prompt",
        type=int,
    )
    parser.add_argument(
        "--generation-tokens",
        "-g",
        default=1024,
        help="Length of completion",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=1,
        help="Batch size",
        type=int,
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        default=5,
        help="Number of timing trials",
        type=int,
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prefill processing (default: 2048)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay between each test in seconds (default: 0)",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="Optional prompt text to tokenize instead of using random token ids.",
    )
    parser.add_argument(
        "--use-block-cache",
        action="store_true",
        help="Enable model-specific block cache when supported.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Block size for models with custom generation.",
    )
    parser.add_argument(
        "--small-block-size",
        type=int,
        default=DEFAULT_SMALL_BLOCK_SIZE,
        help="Sub-block size for models with custom generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold for models with custom generation.",
    )
    parser.add_argument(
        "--mask-id",
        type=int,
        default=None,
        help="Optional mask token id override for models with custom generation.",
    )
    parser.add_argument(
        "--min-unmasks-per-step",
        type=int,
        default=1,
        help="Minimum masked tokens to accept per refinement step for custom generation.",
    )
    return parser


def _build_prompts(tokenizer, prompt_text, prompt_tokens, batch_size, vocab_size):
    if prompt_text is None:
        prompts = mx.random.randint(0, vocab_size, (batch_size, prompt_tokens)).tolist()
        return prompts, False

    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(tokens) == 0:
        raise ValueError("--prompt-text must produce at least one token.")
    if prompt_tokens > 0:
        repeats = (prompt_tokens + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeats)[:prompt_tokens]
    prompts = [list(tokens) for _ in range(batch_size)]
    return prompts, True


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    mx.random.seed(0)

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    model_path = args.model or DEFAULT_MODEL

    if group.size() > 1:
        model, tokenizer, config = sharded_load(
            model_path, pipeline_group, tensor_group, return_config=True
        )
    else:
        model, tokenizer, config = load(
            model_path,
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
            model_config={"quantize_activations": args.quantize_activations},
        )

    # Empty to avoid early stopping
    tokenizer._eos_token_ids = {}

    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens
    batch_size = args.batch_size
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompts, using_text_prompt = _build_prompts(
        tokenizer, args.prompt_text, prompt_tokens, batch_size, vocab_size
    )
    prompt = prompts[0]

    if batch_size > 1 and (
        args.use_block_cache
        or args.block_size != DEFAULT_BLOCK_SIZE
        or args.small_block_size != DEFAULT_SMALL_BLOCK_SIZE
        or args.threshold != DEFAULT_THRESHOLD
        or args.mask_id is not None
        or args.min_unmasks_per_step != DEFAULT_MIN_UNMASKS_PER_STEP
    ):
        raise ValueError(
            "Fast-dLLM generation options are only supported with batch_size=1."
        )

    def single_bench():
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=generation_tokens,
            prefill_step_size=args.prefill_step_size,
            use_block_cache=args.use_block_cache,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            mask_id=args.mask_id,
            min_unmasks_per_step=args.min_unmasks_per_step,
        ):
            pass
        return response

    def batch_bench():
        return batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=generation_tokens,
            prefill_step_size=args.prefill_step_size,
        ).stats

    if batch_size == 1:
        _bench = single_bench
    else:
        _bench = batch_bench

    use_custom_generation_timing = batch_size == 1 and hasattr(
        model, "diffusion_decode"
    )

    def measure_custom_prompt_time():
        prompt_array = mx.array(prompt, dtype=mx.uint32)
        if prompt_array.shape[0] <= args.block_size:
            return 0.0

        full_prefix_len = (prompt_array.shape[0] // args.block_size) * args.block_size
        prefix = prompt_array[None, :full_prefix_len]
        cache = model.make_cache()

        tic = time.perf_counter()
        logits = model(
            prefix,
            cache=cache,
            block_size=args.block_size,
            update_past_key_values=True,
        )
        mx.eval(logits)
        return time.perf_counter() - tic

    rprint("Running warmup..")
    _bench()

    if use_custom_generation_timing:
        rprint(
            "Using wall-clock generation timing for custom model generation."
        )
        if not using_text_prompt:
            rprint(
                "Random-token prompts are pessimistic for confidence-thresholded decoding."
                " Use --prompt-text for a more representative Fast-dLLM benchmark."
            )
        report_keys = ["prompt_tps", "peak_memory"]
    else:
        report_keys = ["prompt_tps", "generation_tps", "peak_memory"]
    rprint(f"Timing with {prompt_tokens=}, {generation_tokens=}, {batch_size=}.")
    responses = []
    total_times = []
    for i in range(args.num_trials):
        if args.delay > 0:
            time.sleep(args.delay)
        prompt_time = measure_custom_prompt_time() if use_custom_generation_timing else None
        tic = time.perf_counter()
        response = _bench()
        toc = time.perf_counter()
        if use_custom_generation_timing:
            generation_time = max(toc - tic - prompt_time, 1e-9)
            response.prompt_tps = (
                response.prompt_tokens / prompt_time if prompt_time > 0 else float("inf")
            )
        else:
            generation_time = None
        responses.append(response)
        total_times.append((toc - tic, generation_time))
        results = [(k, getattr(response, k)) for k in report_keys]
        results = [f"{k}={v:.3f}" for k, v in results]
        if use_custom_generation_timing:
            results.append(
                f"generation_tps_wall={response.generation_tokens / generation_time:.3f}"
            )
        results.append(f"total_time={toc - tic:.3f}")
        rprint(f"Trial {i+1}:  " + ", ".join(results))

    def avg(k):
        vals = (getattr(response, k) for response in responses)
        return sum(vals) / args.num_trials

    results = [(k, avg(k)) for k in report_keys]
    results = [f"{k}={v:.3f}" for k, v in results]
    if use_custom_generation_timing:
        avg_generation_time = max(
            sum(generation_time for _, generation_time in total_times) / args.num_trials,
            1e-9,
        )
        avg_generation_tokens = (
            sum(r.generation_tokens for r in responses) / args.num_trials
        )
        results.append(
            f"generation_tps_wall={avg_generation_tokens / avg_generation_time:.3f}"
        )
    rprint(f"Averages: " + ", ".join(results))


if __name__ == "__main__":
    main()
