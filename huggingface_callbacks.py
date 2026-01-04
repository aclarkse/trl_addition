from transformers import TrainerCallback
import torch
import random
import logging
import wandb
import os


class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.tokenizer = trainer.processing_class
        self._checkpoint_prefix = "checkpoint-"

    def on_save(self, args, state, control, **kwargs):
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(args.output_dir, f"{self._checkpoint_prefix}{state.global_step}")
        if getattr(wandb, "run", None) is not None and getattr(self, "_is_main_process", True): 
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"{self._checkpoint_prefix}{state.global_step}"})
        with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        logging.info("Training ended.")
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(args.output_dir, f"{self._checkpoint_prefix}{state.global_step}")
        if getattr(wandb, "run", None) is not None and getattr(self, "_is_main_process", True): 
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"{self._checkpoint_prefix}{state.global_step}"})
        with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control

class GreedyDecodeOnce(TrainerCallback):
    """
    One-time qualitative probe at training start:
    - Samples a few (default 3) abacus-encoded addition prompts
    - Greedily decodes (manual loop; no .generate()) for max_new_tokens
    - Prints prompt, prediction, target, decoded ints, and correctness

    This replaces the old "The quick brown fox" check, which is meaningless
    for abacus-only training.
    """
    def __init__(
        self,
        trainer,
        tokenizer,
        num_digits: int = 2,
        reverse_digits: bool = True,
        bead: str = "|",
        empty: str = ".",
        sep: str = "/",
        digit_width: int = 9,
        n_problems: int = 3,
        max_new_tokens: int = 64,
        seed: int = 0,
    ):
        self.trainer = trainer
        self.tok = tokenizer

        self.num_digits = int(num_digits)
        self.reverse_digits = bool(reverse_digits)
        self.bead = str(bead)
        self.empty = str(empty)
        self.sep = str(sep)
        self.digit_width = int(digit_width)

        self.n_problems = int(n_problems)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)

        self.done = False

    def _is_main(self, trainer):
        acc = getattr(trainer, "accelerator", None)
        if acc is not None and hasattr(acc, "is_local_main_process"):
            return bool(acc.is_local_main_process)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _encode_digit(self, d: int) -> str:
        d = int(d)
        return (self.bead * d) + (self.empty * (self.digit_width - d))

    def _encode_number(self, n: int, width_digits: int) -> str:
        s = str(int(n)).zfill(width_digits)
        digits = list(s)
        if self.reverse_digits:
            digits = digits[::-1]
        return self.sep.join(self._encode_digit(int(ch)) for ch in digits)

    def _prompt(self, a: int, b: int) -> str:
        return f"{self._encode_number(a, self.num_digits)} + {self._encode_number(b, self.num_digits)} ="

    def _target(self, a: int, b: int) -> str:
        # fixed-length (k+1) digits, matching your dataset design
        return self._encode_number(a + b, self.num_digits + 1)

    def _normalize_abacus(self, text: str) -> str:
        allowed = {self.bead, self.empty, self.sep}
        return "".join(ch for ch in text if ch in allowed)

    def _decode_abacus_to_int(self, abacus: str) -> str:
        """
        Best-effort decode for display. Returns string form, or '?' if malformed.
        """
        abacus = self._normalize_abacus(abacus).strip()
        if not abacus:
            return "?"
        parts = abacus.split(self.sep)
        # If model emits fewer parts than expected, still decode what we can.
        digits = [p.count(self.bead) for p in parts if p]
        if not digits:
            return "?"
        if self.reverse_digits:
            digits = digits[::-1]
        try:
            return str(int("".join(map(str, digits))))
        except Exception:
            return "?"

    def _manual_greedy_decode(self, trainer, prompt: str) -> str:
        model = trainer.model
        device = getattr(trainer.accelerator, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        enc = self.tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if isinstance(attn, torch.Tensor) else torch.ones_like(input_ids, device=device)

        was_training = model.training
        model.eval()

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attn = torch.cat([attn, torch.ones_like(next_token, device=device)], dim=1)

        text = self.tok.decode(input_ids[0], skip_special_tokens=True)
        model.train(was_training)
        return text

    def _run_once(self, trainer):
        if self.done or trainer is None:
            return
        self.done = True

        # sync ranks so we don't interleave prints
        try:
            trainer.accelerator.wait_for_everyone()
        except Exception:
            pass

        if not self._is_main(trainer):
            return

        rng = random.Random(self.seed)

        # sample k-digit a,b (no leading zeros unless k=1)
        if self.num_digits == 1:
            lo, hi = 0, 9
        else:
            lo, hi = 10 ** (self.num_digits - 1), 10 ** self.num_digits - 1

        print("[Abacus Greedy Probe] Running qualitative check on 3 problems...")
        print(f"[Abacus Greedy Probe] k={self.num_digits}, reverse_digits={self.reverse_digits}, digit_width={self.digit_width}")

        for idx in range(self.n_problems):
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)

            prompt = self._prompt(a, b)
            target = self._target(a, b)

            full_text = self._manual_greedy_decode(trainer, prompt)

            # Extract the completion portion by removing the prompt prefix if present
            completion = full_text
            if full_text.startswith(prompt):
                completion = full_text[len(prompt):]

            pred_norm = self._normalize_abacus(completion).strip()
            tgt_norm = self._normalize_abacus(target).strip()

            ok = (pred_norm == tgt_norm)

            print("-" * 80)
            print(f"[#{idx+1}] prompt: {prompt}")
            print(f"[#{idx+1}] pred  : {pred_norm}")
            print(f"[#{idx+1}] target: {tgt_norm}")
            print(f"[#{idx+1}] decoded: {a} + {b} = {a+b}")
            print(f"[#{idx+1}] pred_int: {self._decode_abacus_to_int(pred_norm)}   (match={ok})")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kw):
        self._run_once(getattr(self, "trainer", None))
        return control
