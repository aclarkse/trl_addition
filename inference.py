import json
import os
import logging
import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import hydra
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.dataset import get_datasets


def _coerce_to_str(x, tokenizer=None) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    # chat dict: {"role": ..., "content": ...}
    if isinstance(x, dict) and "content" in x:
        return str(x["content"])
    # chat list: [{"role":"user","content":...}, ...]
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and "content" in x[0]:
        return str(x[0]["content"])
    # pretokenized ids
    if isinstance(x, list) and len(x) > 0 and all(isinstance(t, int) for t in x):
        if tokenizer is None:
            return str(x)
        return tokenizer.decode(x, skip_special_tokens=True)
    return str(x)


def normalize_abacus(text: str, bead: str = "|", empty: str = ".", sep: str = "/") -> str:
    allowed = {bead, empty, sep}
    return "".join(ch for ch in text if ch in allowed).strip()


def decode_abacus_to_int(abacus: str, bead: str = "|", sep: str = "/", reverse_digits: bool = True) -> int:
    abacus = normalize_abacus(abacus, bead=bead, empty=".", sep=sep)
    parts = [p for p in abacus.split(sep) if p != ""]
    digits = [p.count(bead) for p in parts]
    if reverse_digits:
        digits = digits[::-1]
    if not digits:
        raise ValueError("Empty/invalid abacus string")
    return int("".join(str(d) for d in digits))


def parse_prompt_abacus(prompt: str, bead: str = "|", sep: str = "/", reverse_digits: bool = True) -> Tuple[int, int]:
    # Prompt format: "<abacus(a)> + <abacus(b)> ="
    if "+" not in prompt:
        raise ValueError("Prompt missing '+'")
    left, rest = prompt.split("+", 1)
    left = left.strip()
    right = rest.replace("=", " ").strip()

    a = decode_abacus_to_int(left, bead=bead, sep=sep, reverse_digits=reverse_digits)
    b = decode_abacus_to_int(right, bead=bead, sep=sep, reverse_digits=reverse_digits)
    return a, b


def count_carries(a: int, b: int, k: int) -> int:
    carries = 0
    carry = 0
    for _ in range(k):
        da = a % 10
        db = b % 10
        s = da + db + carry
        if s >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return carries


def score_em_by_carries(
    examples: List[Dict[str, Any]],
    bead: str,
    empty: str,
    sep: str,
    reverse_digits: bool,
    k: int,
) -> Dict[str, Any]:
    total = 0
    correct_any = 0

    slice_stats = {
        "0": {"n": 0, "correct_any": 0},
        "1": {"n": 0, "correct_any": 0},
        "2plus": {"n": 0, "correct_any": 0},
    }

    failures: List[Dict[str, str]] = []

    for ex in examples:
        prompt = ex["prompt"]
        target = ex["ground_truth"][-1]
        target_norm = normalize_abacus(target, bead=bead, empty=empty, sep=sep)

        preds = ex["generated_text"]
        preds_norm = [normalize_abacus(p, bead=bead, empty=empty, sep=sep) for p in preds]

        is_correct = any(p == target_norm for p in preds_norm)

        total += 1
        correct_any += int(is_correct)

        # carry bucket
        try:
            a, b = parse_prompt_abacus(prompt, bead=bead, sep=sep, reverse_digits=reverse_digits)
            c = count_carries(a, b, k) if k >= 2 else 0
            bucket = "2plus" if c >= 2 else str(c)
        except Exception:
            bucket = "0"

        slice_stats[bucket]["n"] += 1
        slice_stats[bucket]["correct_any"] += int(is_correct)

        if (not is_correct) and len(failures) < 25:
            failures.append(
                {
                    "prompt": prompt,
                    "target_norm": target_norm,
                    "pred_norm_0": preds_norm[0] if preds_norm else "",
                }
            )

    return {
        "n": total,
        "exact_match": correct_any / max(1, total),
        "by_carries": {
            key: (slice_stats[key]["correct_any"] / max(1, slice_stats[key]["n"]))
            for key in slice_stats
        },
        "failures_sample": failures,
    }

@dataclass
class GenConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    num_samples: int = 1
    batch_size: int = 64


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Any],
    gen_cfg: GenConfig,
) -> List[List[str]]:
    device = model.device

    prompts = [_coerce_to_str(p, tokenizer=tokenizer) for p in prompts]
    # Keep alignment; if something becomes empty, replace with a safe dummy prompt (rare)
    prompts = [p if (isinstance(p, str) and len(p) > 0) else "." for p in prompts]

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else None

    do_sample = (gen_cfg.temperature is not None) and (gen_cfg.temperature > 0.0) and (gen_cfg.num_samples > 1)
    num_return_sequences = gen_cfg.num_samples if do_sample else 1

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=int(gen_cfg.max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(gen_cfg.temperature) if do_sample else 1.0,
        num_return_sequences=int(num_return_sequences),
        pad_token_id=tokenizer.eos_token_id,
    )

    bsz = input_ids.shape[0]
    in_len = input_ids.shape[1]

    decoded = tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)

    if not do_sample:
        return [[d] for d in decoded]

    grouped: List[List[str]] = []
    for i in range(bsz):
        start = i * gen_cfg.num_samples
        grouped.append(decoded[start:start + gen_cfg.num_samples])
    return grouped


def run_inference_split(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ds,
    split_name: str,
    gen_cfg: GenConfig,
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    n = len(ds)
    if max_examples is not None:
        n = min(n, int(max_examples))
        ds = ds.select(range(n))

    outputs: List[Dict[str, Any]] = []
    num_batches = math.ceil(n / gen_cfg.batch_size)
    logging.info(f"[{split_name}] Inference on {n} examples in {num_batches} batches...")

    for bi in range(num_batches):
        start = bi * gen_cfg.batch_size
        end = min(n, (bi + 1) * gen_cfg.batch_size)
        batch = ds.select(range(start, end))

        # IMPORTANT: force python lists (avoid HF Column objects)
        prompts = list(batch["prompt"])
        example_ids = list(batch["example_id"])
        ground_truth = list(batch["ground_truth"])

        gens = generate_batch(model, tokenizer, prompts, gen_cfg)

        outputs.append({
            "prompt": prompts,
            "generated_text": gens,
            "example_id": example_ids,
            "ground_truth": ground_truth,
        })

    return outputs


def flatten_generated_outputs(raw_batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for batch in raw_batches:
        prompts = batch["prompt"]
        gens = batch["generated_text"]
        eids = batch["example_id"]
        gts = batch["ground_truth"]
        for i in range(len(prompts)):
            flat.append({
                "prompt": prompts[i],
                "generated_text": gens[i],
                "example_id": eids[i],
                "ground_truth": gts[i],
            })
    return flat

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args):
    logging.basicConfig(level=logging.INFO)

    model_dir = str(getattr(args.model_pars, "model_dir", "") or "")
    hf_model_id = str(getattr(args.model_pars, "hf_model_id", "gpt2"))
    hf_tokenizer_id = str(getattr(args.model_pars, "hf_tokenizer_id", hf_model_id))

    load_id = model_dir if model_dir else hf_model_id
    tok_id = model_dir if model_dir else hf_tokenizer_id

    logging.info(f"Loading tokenizer: {tok_id}")
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Loading model: {load_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(load_id).to(device)
    model.eval()

    # datasets
    train_ds, test_ds = get_datasets(args, tokenizer)

    # eval sizing
    max_eval_train = int(getattr(args.eval_pars, "max_eval_train", 2000))
    max_eval_test = int(getattr(args.eval_pars, "max_eval_test", 5000))

    # generation config
    gen_cfg = GenConfig(
        max_new_tokens=int(getattr(args.eval_pars, "max_tokens", 64)),
        temperature=float(getattr(args.eval_pars, "temperature", 0.0)),
        num_samples=int(getattr(args.eval_pars, "num_samples", 1)),
        batch_size=int(getattr(args.eval_pars, "batch_size", 64)),
    )

    # scoring params
    bead = str(getattr(args.dataset_pars, "bead", "|"))
    empty = str(getattr(args.dataset_pars, "empty", "."))
    sep = "/"
    reverse_digits = bool(getattr(args.dataset_pars, "reverse_digits", True))
    k = int(getattr(args.dataset_pars, "num_digits", 2))

    # output dir (default: model_dir if provided else cwd)
    out_dir = model_dir if model_dir else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # inference
    train_batches = run_inference_split(model, tokenizer, train_ds, "train", gen_cfg, max_examples=max_eval_train)
    test_batches = run_inference_split(model, tokenizer, test_ds, "test", gen_cfg, max_examples=max_eval_test)

    train_examples = flatten_generated_outputs(train_batches)
    test_examples = flatten_generated_outputs(test_batches)

    train_metrics = score_em_by_carries(train_examples, bead=bead, empty=empty, sep=sep, reverse_digits=reverse_digits, k=k)
    test_metrics = score_em_by_carries(test_examples, bead=bead, empty=empty, sep=sep, reverse_digits=reverse_digits, k=k)

    # print
    print("\n================= RESULTS =================")
    print(f"[TRAIN] n={train_metrics['n']}  EM={train_metrics['exact_match']:.4f}  carries={train_metrics['by_carries']}")
    print(f"[TEST ] n={test_metrics['n']}  EM={test_metrics['exact_match']:.4f}  carries={test_metrics['by_carries']}")
    print("==========================================\n")

    metrics_path = os.path.join(out_dir, "metrics_train_test.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "k": k,
                    "reverse_digits": reverse_digits,
                    "bead": bead,
                    "empty": empty,
                    "sep": sep,
                    "gen_cfg": asdict(gen_cfg),
                    "train_scored": int(train_metrics["n"]),
                    "test_scored": int(test_metrics["n"]),
                },
                "train": train_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    logging.info(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
