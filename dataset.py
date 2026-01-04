from typing import Literal, Optional
from datasets import Dataset
from collections import defaultdict
import random
import regex
import json


def num_carry_ops_k(a: int, b: int, k: int) -> int:
    """Count carry operations for base-10 addition, considering exactly k digits."""
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


class AbacusAdditionDataset:
    def __init__(
        self,
        seed: int,
        tokenizer,
        num_digits: int = 3,
        n_train: int = 20000,
        n_test: int = 2000,
        reverse_digits: bool = True,
        commutative_aug: bool = True,

        bead: str = "|",
        empty: str = ".",
        digit_width: int = 9,
        digit_bias_mode: Literal["uniform", "large"] = "uniform",
        p_large_digit: float = 0.7, # prob of sampling a large digit when digit_bias_mode="large"
        min_num_carries_train: Optional[int] = None,
        min_num_carries_test: Optional[int] = None
    ):
        self.seed = seed
        self.tokenizer = tokenizer
        self.num_digits = int(num_digits)
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.reverse_digits = bool(reverse_digits)
        self.commutative_aug = bool(commutative_aug)

        self.bead = bead
        self.empty = empty
        self.digit_width = int(digit_width)
        self.digit_bias_mode = digit_bias_mode
        self.p_large_digit = float(p_large_digit)
        self.min_num_carries_train = min_num_carries_train
        self.min_num_carries_test = min_num_carries_test

        if digit_bias_mode not in ("uniform", "large"):
            raise ValueError(f"Unknown digit_bias_mode: {digit_bias_mode}")
   

    def _encode_digit_abacus(self, d: int) -> str:
        d = int(d)
        assert 0 <= d <= 9
        return (self.bead * d) + (self.empty * (self.digit_width - d))

    def _encode_number(self, n: int, width_digits: int) -> str:
        """
        Encode n as exactly `width_digits` base-10 digits, then abacus-encode each digit.
        If reverse_digits=True, digits are reversed (ones-first).
        Delimiter between digits: "/".
        """
        s = str(int(n)).zfill(width_digits)
        digits = list(s)
        if self.reverse_digits:
            digits = digits[::-1]
        enc_digits = [self._encode_digit_abacus(int(ch)) for ch in digits]
        return "/".join(enc_digits)

    def _prompt(self, a: int, b: int) -> str:
        return f"{self._encode_number(a, self.num_digits)} + {self._encode_number(b, self.num_digits)} ="

    def _answer(self, a: int, b: int) -> str:
        s = a + b
        return self._encode_number(s, self.num_digits + 1)

    
    def _sample_digit(self, rng: random.Random) -> int:
        if getattr(self, "digit_bias_mode", "uniform") == "large":
            if rng.random() < self.p_large_digit:
                return rng.choice([8, 9])
            return rng.randint(0, 7)
        return rng.randint(0, 9)

    def _sample_number(self, rng: random.Random) -> int:
        digits = [self._sample_digit(rng) for _ in range(self.num_digits)]
        if self.num_digits > 1 and digits[-1] == 0:
            digits[-1] = rng.randint(1, 9) if self.digit_bias_mode != "large" else rng.choice([8, 9])
        n = 0
        for i, d in enumerate(digits):
            n += d * (10 ** i)
        return n

    def _sample_pair(self, rng: random.Random) -> tuple[int, int]:
        return self._sample_number(rng), self._sample_number(rng)
    

    def _fill_split(self, n_examples: int, rng: random.Random, split: str):
        """
        Return list of (a,b,num_carries) pairs for this split.
        Carry difficulty is controlled by:
        - digit_bias_mode / p_large_digit
        - min_num_carries_train / min_num_carries_test
        """
        pairs = []
        min_carries = (
            self.min_num_carries_train if split == "train"
            else self.min_num_carries_test
        )

        while len(pairs) < n_examples:
            a, b = self._sample_pair(rng)
            c = num_carry_ops_k(a, b, self.num_digits)

            if min_carries is not None and c < min_carries:
                continue

            pairs.append((a, b, c))

        return pairs

    def generate_data(self):
        rng = random.Random(self.seed)

        train_pairs = self._fill_split(self.n_train, rng, split="train")
        test_pairs  = self._fill_split(self.n_test,  rng, split="test")

        def build_dataset(pairs):
            ds = {"prompt": [], "completion": [], "example_id": [], "ground_truth": []}
            example_id = 0

            for (a, b, _carries) in pairs:
                prompt = self._prompt(a, b)
                ans = self._answer(a, b)

                ds["prompt"].append(prompt)
                ds["completion"].append(ans)
                ds["example_id"].append(example_id)
                ds["ground_truth"].append([ans])  # keep list so ground_truth[-1] works everywhere
                example_id += 1

                # commutativity augmentation
                if self.commutative_aug and a != b:
                    prompt2 = self._prompt(b, a)
                    ds["prompt"].append(prompt2)
                    ds["completion"].append(ans)
                    ds["example_id"].append(example_id)
                    ds["ground_truth"].append([ans])
                    example_id += 1

            hf = Dataset.from_dict(ds).shuffle(seed=self.seed)
            return hf

        return build_dataset(train_pairs), build_dataset(test_pairs)


    def save_state(self, save_path):
        state = {
            "seed": self.seed,
            "num_digits": self.num_digits,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "reverse_digits": self.reverse_digits,
            "commutative_aug": self.commutative_aug,
            "bead": self.bead,
            "empty": self.empty,
            "digit_width": self.digit_width,
            "digit_bias_mode": self.digit_bias_mode,
            "p_large_digit": self.p_large_digit,
            "min_num_carries_train": self.min_num_carries_train,
            "min_num_carries_test": self.min_num_carries_test
        }
        with open(save_path, "w") as f:
            json.dump(state, f)

    @classmethod
    def from_state(cls, state_path, tokenizer):
        with open(state_path, "r") as f:
            state = json.load(f)
        return cls(tokenizer=tokenizer, **state)


def math_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_datasets(args, tokenizer):
    """
    train.py calls this: train_dataset, eval_dataset = get_datasets(args, tokenizer)
    """
    p = args.dataset_pars
    ds = AbacusAdditionDataset(
        seed=int(args.seed),
        tokenizer=tokenizer,
        num_digits=int(getattr(p, "num_digits", 2)),
        n_train=int(getattr(p, "n_train", 20000)),
        n_test=int(getattr(p, "n_test", 2000)),
        reverse_digits=bool(getattr(p, "reverse_digits", True)),
        commutative_aug=bool(getattr(p, "commutative_aug", True)),
        bead=str(getattr(p, "bead", "|")),
        empty=str(getattr(p, "empty", ".")),
        digit_width=int(getattr(p, "digit_width", 9)),
        digit_bias_mode=str(getattr(p, "digit_bias_mode", "uniform")),
        p_large_digit=float(getattr(p, "p_large_digit", 0.7)),
        min_num_carries_train=getattr(p, "min_num_carries_train", None),
        min_num_carries_test=getattr(p, "min_num_carries_test", None)
    )
    return ds.generate_data()


def get_dataset(args, tokenizer):
    """
    inference.py calls this: dataset = get_dataset(args, tokenizer)
    We'll return the *test* dataset by default (more sensible for eval).
    """
    train_ds, test_ds = get_datasets(args, tokenizer)
    return test_ds
