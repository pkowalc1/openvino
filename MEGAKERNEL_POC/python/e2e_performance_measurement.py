"""End-to-end MegaKernel decode performance measurement.

Usage
-----
    source /opt/home/pwysocki/openvino_dist/setupvars.sh
    /opt/home/pwysocki/.venv/bin/python e2e_performance_measurement.py
    ... --device GPU.1 --frameworks decode_only optimum genai
    ... --tokens 200
    ... --only-framework decode_only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_MODEL_DIR = HERE / "qwen3-0.6b-openvino-ir"
DEFAULT_DEVICE = "GPU.1"
BATCH = 1

DECODE_ONLY_PROMPT = "What is the capital of France?"

PROMPTS: list[dict[str, str]] = [
    {
        "name": "short",
        "text": "What is the capital of France?",
    },
    {
        "name": "medium",
        "text": (
            "Explain, in a few sentences, how a transformer neural network uses "
            "self-attention to process a sequence of tokens, and why key-value "
            "caching makes autoregressive decoding faster than recomputing the "
            "whole sequence at every step."
        ),
    },
    {
        "name": "long",
        "text": (
            "You are a senior systems engineer. Read the following background "
            "carefully and then answer the question at the end.\n\n"
            "Large language models are deployed on a wide range of hardware, from "
            "small integrated GPUs to large data-center accelerators. During "
            "inference the model first runs a prefill phase that processes the "
            "entire prompt in a single forward pass, populating the key-value "
            "cache for every attention layer. After prefill the model enters the "
            "decode phase, generating one token at a time. Each decode step reads "
            "the growing key-value cache, computes attention against all previous "
            "tokens, and appends the new key and value vectors. Because the decode "
            "phase is memory-bandwidth bound and launches many small kernels, it "
            "often dominates end-to-end latency for long generations. A megakernel "
            "fuses the many small per-layer kernels of a decode step into a single "
            "GPU kernel launch, preloading weights for the next operation while the "
            "current one computes, using fine-grained synchronization, and removing "
            "kernel launch overhead and tail effects. This is particularly valuable "
            "for small models on small GPUs where launch overhead is a large "
            "fraction of the total step time.\n\n"
            "Question: Given the description above, explain why fusing the decode "
            "step into a single megakernel is expected to improve latency more than "
            "it improves prefill, and describe one hardware limitation that could "
            "reduce the achievable speedup on a small GPU."
        ),
    },
    {
        "name": "v. long",
        "text": (
            200 * "You are a senior systems engineer. Read the following background and talk about CPU cache "
        ),
    },
]


def get_prompts(args) -> list[dict[str, str]]:
    """Prompt set for optimum/genai: a single user-supplied prompt when --prompt is
    given, otherwise the built-in short/medium/long trio."""
    custom = getattr(args, "prompt", None)
    if custom:
        return [{"name": "custom", "text": custom}]
    return PROMPTS


# ---------------------------------------------------------------------------
# OV inference helpers
# ---------------------------------------------------------------------------

def prefill_inputs(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    seq_len = input_ids.shape[1]
    return {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": np.ones((BATCH, seq_len), np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
        "beam_idx": np.zeros(BATCH, np.int32),
    }


def single_token_inputs(token_id: int, position: int) -> dict[str, np.ndarray]:
    """One-token step (decode, or one step of token-by-token priming)."""
    return {
        "input_ids": np.array([[token_id]], np.int64),
        "attention_mask": np.ones((BATCH, position + 1), np.int64),
        "position_ids": np.array([[position]], np.int64),
        "beam_idx": np.zeros(BATCH, np.int32),
    }


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def prompt_token_ids(tokenizer, prompt: str) -> np.ndarray:
    text = chat_text(tokenizer, prompt)
    ids = tokenizer([text], return_tensors="np").input_ids
    return ids.astype(np.int64)


def stats(latencies_ms: list[float]) -> dict[str, float]:
    lat = sorted(latencies_ms)
    n = len(lat)
    mean = statistics.mean(lat)
    return {
        "mean": mean,
        "median": lat[n // 2],
        "min": lat[0],
        "p90": lat[int(0.90 * (n - 1))],
        "p99": lat[int(0.99 * (n - 1))],
        "tok_s": 1000.0 / mean,
        "count": n,
    }


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def decode_only_worker(args) -> list[dict]:
    """Pure decode benchmark using the OV native API.

    Prefill is run once (untimed) to warm the KV cache, then N identical
    single-token decode steps are timed at a fixed position.  No prefill
    latency is measured or reported.  A single fixed prompt is used; its
    content does not affect the measured decode cost.
    """
    import openvino as ov

    core = ov.Core()
    dev_name = core.get_property(args.device, "FULL_DEVICE_NAME")
    model = core.read_model(str(Path(args.model_dir) / "openvino_model.xml"))
    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device)
    compile_s = time.perf_counter() - t0

    tokenizer = load_tokenizer(Path(args.model_dir))

    # Warmup – dummy prefill + a few decode steps
    warm = compiled.create_infer_request()
    warm.infer(prefill_inputs(np.ones((BATCH, 8), np.int64)))
    for pos in range(8, 12):
        warm.infer(single_token_inputs(1, pos))

    # Tokenize the fixed prompt (content is irrelevant to the timed section)
    ids = prompt_token_ids(tokenizer, DECODE_ONLY_PROMPT)
    prompt_len = int(ids.shape[1])

    # Prefill to populate the KV cache (untimed)
    req = compiled.create_infer_request()
    res = req.infer(prefill_inputs(ids))
    logits = np.array(res[0])[0, -1, :].astype(np.float32)
    next_id = int(logits.argmax())

    # Optionally prime the KV cache to a longer context
    target_ctx = max(args.decode_ctx, prompt_len)
    for priming_pos in range(prompt_len, target_ctx):
        req.infer(single_token_inputs(next_id, priming_pos))

    decode_pos = target_ctx
    # The same decode input is reused every iteration to isolate kernel cost
    decode_input = single_token_inputs(next_id, decode_pos)

    for _ in range(args.warmup):
        req.infer(decode_input)
    lat = []
    for _ in range(args.tokens):
        t = time.perf_counter()
        req.infer(decode_input)
        lat.append((time.perf_counter() - t) * 1e3)

    # Greedy-decode real text for baseline vs megakernel comparison (untimed,
    # fresh request so the timed loop above is unaffected).
    text_out = _native_generate_text(compiled, tokenizer, ids, prompt_len, args.tokens)

    return [{
        "prompt": "decode",
        "prompt_len": prompt_len,
        "decode_ctx": decode_pos,
        "n_tok": args.tokens,
        "decode": stats(lat),
        "argmax": next_id,
        "logits": logits.tolist(),
        "device": dev_name,
        "compile_s": compile_s,
        "text": text_out,
    }]


def _native_generate_text(compiled, tokenizer, ids: np.ndarray, prompt_len: int,
                           n_tokens: int) -> str:
    """Greedy-decode real tokens from a fresh request and detokenize."""
    gen_req = compiled.create_infer_request()
    r = gen_req.infer(prefill_inputs(ids))
    cur = int(np.array(r[0])[0, -1, :].argmax())
    pos = prompt_len
    eos = tokenizer.eos_token_id
    gen_ids: list[int] = []
    for _ in range(n_tokens):
        gen_ids.append(cur)
        if eos is not None and cur == eos:
            break
        r = gen_req.infer(single_token_inputs(cur, pos))
        pos += 1
        cur = int(np.array(r[0])[0, -1, :].argmax())
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def optimum_worker(args) -> list[dict]:
    import torch
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    # Cap torch's intra-op threads. optimum's generate() runs its sampling/
    # logits-processing (torch isin/where/arange, all multi-threaded) on the host
    # BETWEEN forwards, while OpenVINO's async GPU infer path keeps host threads
    # busy. With torch defaulting to one thread per core, those gap ops
    # oversubscribe the CPU and get throttled 2-3x -- which erases the
    # MegaKernel's faster inference and makes decode look no faster than baseline.
    # Leaving CPU headroom removes the oversubscription and restores the speedup.
    n_threads = args.torch_threads or max(1, (os.cpu_count() or 8) // 4)
    torch.set_num_threads(n_threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    # CACHE_DIR="" disables the compiled-model blob cache. The cache key does not
    # include the OV_MEGAKERNEL_DISABLE env var, so leaving it on would make the
    # MegaKernel run silently reuse the baseline blob (no transformation).
    model = OVModelForCausalLM.from_pretrained(
        args.model_dir, device=args.device, ov_config={"CACHE_DIR": ""})
    compile_s = time.perf_counter() - t0

    import openvino as ov
    dev_name = ov.Core().get_property(args.device, "FULL_DEVICE_NAME")

    def gen(model_inputs, n_new):
        t = time.perf_counter()
        out = model.generate(
            **model_inputs,
            max_new_tokens=n_new,
            min_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
        )
        return (time.perf_counter() - t) * 1e3, out

    results = []
    for prompt in get_prompts(args):
        text = chat_text(tokenizer, prompt["text"])
        model_inputs = tokenizer([text], return_tensors="pt")
        prompt_len = int(model_inputs.input_ids.shape[1])

        # warmup
        for _ in range(max(1, args.gen_warmup)):
            gen(model_inputs, args.tokens)

        # TTFT (prefill only) = generate exactly one new token.
        ttft = []
        for _ in range(args.gen_iters):
            ms, _ = gen(model_inputs, 1)
            ttft.append(ms)
        # Full generation of args.tokens tokens.
        full = []
        last_out = None
        for _ in range(args.gen_iters):
            ms, last_out = gen(model_inputs, args.tokens)
            full.append(ms)

        ttft_mean = statistics.mean(ttft)
        full_mean = statistics.mean(full)
        out_ids = last_out[0][prompt_len:].tolist()
        actual_n_tok = len(out_ids)
        # TTFT measures 1-token generation; remaining (actual_n_tok - 1) tokens
        # are decode steps. Guard against degenerate short outputs.
        n_decode = max(actual_n_tok - 1, 1)
        decode_total = max(full_mean - ttft_mean, 1e-6)
        per_tok_ms = decode_total / n_decode
        gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)

        results.append({
            "prompt": prompt["name"],
            "prompt_len": prompt_len,
            "prefill_ms": ttft_mean,
            "n_tok": actual_n_tok,
            "decode": {
                "mean": per_tok_ms,
                "median": per_tok_ms,
                "tok_s": 1000.0 / per_tok_ms,
                "count": n_decode,
            },
            "device": dev_name,
            "compile_s": compile_s,
            "text": gen_text,
        })
    return results


def genai_worker(args) -> list[dict]:
    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    # CACHE_DIR="" disables the compiled-model blob cache (see optimum_worker).
    # Baseline uses GenAI's default PagedAttention backend.
    pipeline_kwargs: dict = {"CACHE_DIR": ""}
    pipeline_kwargs["ATTENTION_BACKEND"] = "SDPA"
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device, **pipeline_kwargs)
    compile_s = time.perf_counter() - t0

    import openvino as ov
    dev_name = ov.Core().get_property(args.device, "FULL_DEVICE_NAME")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.tokens
    cfg.min_new_tokens = args.tokens
    cfg.do_sample = False
    cfg.num_beams = 1

    results = []
    for prompt in get_prompts(args):
        text = chat_text(tokenizer, prompt["text"])
        prompt_len = int(tokenizer([text], return_tensors="np").input_ids.shape[1])

        for _ in range(max(1, args.gen_warmup)):
            pipe.generate([text], cfg)

        ttft_ms, tpot_ms, tput = [], [], []
        gen_text = ""
        for _ in range(args.gen_iters):
            res = pipe.generate([text], cfg)
            pm = res.perf_metrics
            ttft_ms.append(pm.get_ttft().mean)
            tpot_ms.append(pm.get_tpot().mean)     # decode: mean ms / output token
            tput.append(pm.get_throughput().mean)
            gen_text = res.texts[0] if getattr(res, "texts", None) else str(res)

        tpot_mean = statistics.mean(tpot_ms)
        actual_n_tok = res.perf_metrics.get_num_generated_tokens()
        results.append({
            "prompt": prompt["name"],
            "prompt_len": prompt_len,
            "prefill_ms": statistics.mean(ttft_ms),
            "n_tok": actual_n_tok,
            "decode": {
                "mean": tpot_mean,
                "median": tpot_mean,
                "tok_s": 1000.0 / tpot_mean,
                "throughput_tok_s": statistics.mean(tput),
                "count": actual_n_tok,
            },
            "device": dev_name,
            "compile_s": compile_s,
            "text": gen_text,
        })
    return results


WORKERS = {
    "decode_only": decode_only_worker,
    "optimum": optimum_worker,
    "genai": genai_worker,
}


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def worker_env(framework: str, path: str) -> dict:
    env = os.environ.copy()
    env["OV_MEGAKERNEL_DISABLE"] = "0" if path == "megakernel" else "1"
    return env


def spawn(framework: str, path: str, args) -> list[dict]:
    cmd = [
        sys.executable, __file__, "--worker", framework, "--path", path,
        "--model-dir", str(args.model_dir), "--device", args.device,
        "--warmup", str(args.warmup), "--tokens", str(args.tokens),
        "--decode-ctx", str(args.decode_ctx),
        "--torch-threads", str(args.torch_threads),
        "--gen-warmup", str(args.gen_warmup), "--gen-iters", str(args.gen_iters),
    ]
    if getattr(args, "prompt", None):
        cmd += ["--prompt", args.prompt]
    out = subprocess.run(cmd, env=worker_env(framework, path), capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        raise RuntimeError(f"{framework}/{path} worker failed (exit {out.returncode})")
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    for ln in lines[:-1]:
        print(f"    [{framework}/{path}] {ln}")
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def _maybe_print_text(base: list[dict], mega: list[dict]) -> None:
    """Print generated text only when baseline and megakernel outputs differ."""
    for b, m in zip(base, mega):
        bt = (b.get("text") or "").strip()
        mt = (m.get("text") or "").strip()
        if bt != mt:
            print(f"  *** OUTPUT MISMATCH [{b['prompt']}] ***")
            print(f"    baseline  : {bt!r}")
            print(f"    megakernel: {mt!r}")


def print_decode_only_table(base: list[dict], mega: list[dict]) -> None:
    W = 86
    print()
    print("=" * W)
    print(" DECODE-ONLY  (OV native API)")
    print("=" * W)
    print(" How it works:")
    print("   Prefill is executed once (untimed) to warm the KV cache.")
    print("   Then N identical single-token decode steps are timed at a fixed")
    print("   KV-cache position.  No prefill latency is measured or reported.")
    print("   Prompt content does not affect the measured decode cost.")
    print("   decode_x is the per-token decode speedup (primary metric).")
    print()

    hdr = (f"  {'ctx':>5}  {'n_tok':>5} | "
           f"{'base ms/tok':>11}  {'base tok/s':>10} | "
           f"{'mk ms/tok':>9}  {'mk tok/s':>8} | "
           f"{'decode_x':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        b_toks = b["decode"]["tok_s"]
        m_toks = m["decode"]["tok_s"]
        dec_x = bd / md if md else float("nan")
        extra = ""
        if "logits" in b and "logits" in m:
            match = b["argmax"] == m["argmax"]
            cos = cosine(b["logits"], m["logits"])
            extra = f"   argmatch={match}  cos={cos:.4f}"
        print(f"  {b['decode_ctx']:>5}  {b['n_tok']:>5} | "
              f"{bd:>11.3f}  {b_toks:>10.1f} | "
              f"{md:>9.3f}  {m_toks:>8.1f} | "
              f"{dec_x:>7.2f}x{extra}")

    _maybe_print_text(base, mega)


def print_optimum_table(base: list[dict], mega: list[dict], n_tokens: int) -> None:
    W = 98
    print()
    print("=" * W)
    print(" OPTIMUM-INTEL  (HF OVModelForCausalLM.generate)")
    print("=" * W)
    print(" How it works:")
    print("   TTFT = time to first token (prefill latency), measured by generating")
    print("   exactly 1 new token.  ms/tok = per-decode-token latency estimated as")
    print("   (full_generate_ms - ttft_ms) / (n_generated - 1), averaged over")
    print("   multiple generate() calls.  decode_x is the primary speedup metric.")
    print()

    hdr = (f"  {'prompt':<8}  {'in_tok':>6} | "
           f"{'base ttft':>9}  {'mk ttft':>7} | "
           f"{'base ms/tok':>11}  {'mk ms/tok':>9} | "
           f"{'decode_x':>8}  {'pf_x':>5}  {f'e2e_x@{n_tokens}':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        bpf = b.get("prefill_ms", float("nan"))
        mpf = m.get("prefill_ms", float("nan"))
        dec_x = bd / md if md else float("nan")
        pf_x = bpf / mpf if mpf else float("nan")
        n_dec = max(n_tokens - 1, 0)
        e2e = (bpf + n_dec * bd) / (mpf + n_dec * md) if mpf and md else float("nan")
        print(f"  {b['prompt']:<8}  {b['prompt_len']:>6} | "
              f"{bpf:>9.3f}  {mpf:>7.3f} | "
              f"{bd:>11.3f}  {md:>9.3f} | "
              f"{dec_x:>7.2f}x  {pf_x:>4.2f}x  {e2e:>9.2f}x")

    _maybe_print_text(base, mega)


def print_genai_table(base: list[dict], mega: list[dict], n_tokens: int) -> None:
    W = 98
    print()
    print("=" * W)
    print(" OPENVINO GENAI  (ov_genai.LLMPipeline.generate)")
    print("=" * W)
    print(" How it works:")
    print("   Baseline path uses the PagedAttention backend.")
    print("   TTFT and TPOT come from ov_genai perf_metrics, averaged over")
    print("   multiple generate() calls.  decode_x is the primary speedup metric.")
    print("   pf_x is NOT an apples-to-apples backend comparison: the megakernel")
    print("   pipeline is forced onto ATTENTION_BACKEND=SDPA (InsertMegaKernel only")
    print("   pattern-matches the per-layer ReadValue/Assign KV state used by SDPA,")
    print("   not PagedAttention's fused kv-cache), so its prefill runs the plain")
    print("   SDPA reference kernel while baseline gets PagedAttention's optimized,")
    print("   length-scaling prefill. A low/negative pf_x here reflects that backend")
    print("   gap, not megakernel overhead.")
    print()

    hdr = (f"  {'prompt':<8}  {'in_tok':>6} | "
           f"{'base ttft':>9}  {'mk ttft':>7} | "
           f"{'base ms/tok':>11}  {'mk ms/tok':>9} | "
           f"{'decode_x':>8}  {'pf_x':>5}  {f'e2e_x@{n_tokens}':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        bpf = b.get("prefill_ms", float("nan"))
        mpf = m.get("prefill_ms", float("nan"))
        dec_x = bd / md if md else float("nan")
        pf_x = bpf / mpf if mpf else float("nan")
        n_dec = max(n_tokens - 1, 0)
        e2e = (bpf + n_dec * bd) / (mpf + n_dec * md) if mpf and md else float("nan")
        print(f"  {b['prompt']:<8}  {b['prompt_len']:>6} | "
              f"{bpf:>9.3f}  {mpf:>7.3f} | "
              f"{bd:>11.3f}  {md:>9.3f} | "
              f"{dec_x:>7.2f}x  {pf_x:>4.2f}x  {e2e:>9.2f}x")

    _maybe_print_text(base, mega)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="GPU.1 = B60 dGPU")
    ap.add_argument("--frameworks", nargs="+", default=["decode_only", "optimum", "genai"],
                    choices=list(WORKERS))
    ap.add_argument("--only-framework", choices=list(WORKERS), default=None,
                    help="Run a single framework (overrides --frameworks).")
    ap.add_argument("--tokens", type=int, default=150,
                    help="Decode steps to time (decode_only) / tokens to generate "
                         "(optimum, genai) per path.")
    # decode_only tuning
    ap.add_argument("--warmup", type=int, default=5,
                    help="Decode-step warmup iterations before timing (decode_only only).")
    ap.add_argument("--decode-ctx", type=int, default=0,
                    help="If >0, prime the KV cache to this length before timing decode "
                         "(isolates kernel cost from O(context) attention growth). "
                         "decode_only path only.")
    # optimum / genai generate() benchmark
    ap.add_argument("--torch-threads", type=int, default=23,
                    help="Cap torch intra-op threads in the optimum path "
                         "(0 = auto: ~cores/4).")
    ap.add_argument("--gen-warmup", type=int, default=1,
                    help="generate() warmup calls (optimum/genai paths).")
    ap.add_argument("--gen-iters", type=int, default=3,
                    help="generate() measured calls (optimum/genai paths).")
    ap.add_argument("--prompt", default=None,
                    help="Single custom prompt for optimum/genai paths (overrides the "
                         "built-in short/medium/long trio). Has no effect on decode_only.")
    # internal
    ap.add_argument("--worker", choices=list(WORKERS), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--path", choices=("baseline", "megakernel"), default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        args.model_dir = str(args.model_dir)
        results = WORKERS[args.worker](args)
        print(json.dumps(results))
        return

    frameworks = [args.only_framework] if args.only_framework else args.frameworks
    print(f"Device: {args.device}   frameworks: {frameworks}")
    print(f"tokens={args.tokens}   decode_only warmup={args.warmup}   "
          f"generate: warmup={args.gen_warmup} iters={args.gen_iters}")

    all_results: dict[str, dict[str, list[dict]]] = {}
    for fw in frameworks:
        all_results[fw] = {}
        for path in ("baseline", "megakernel"):
            print(f"\n>>> running {fw}/{path} ...")
            all_results[fw][path] = spawn(fw, path, args)

    for fw in frameworks:
        base = all_results[fw]["baseline"]
        mega = all_results[fw]["megakernel"]
        if fw == "decode_only":
            print_decode_only_table(base, mega)
        elif fw == "optimum":
            print_optimum_table(base, mega, args.tokens)
        elif fw == "genai":
            print_genai_table(base, mega, args.tokens)

    print()


if __name__ == "__main__":
    main()
