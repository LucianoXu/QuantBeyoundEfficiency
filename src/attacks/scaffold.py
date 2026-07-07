
from typing import Any
import random

import torch


def make_affirmative_target(instr: str, max_words: int = 20) -> str:
    '''
    Construct the customized target according to the instruction.

    Same with GCG paper.
    '''

    # normalize
    s = " ".join(instr.strip().split()).rstrip(".!?")
    if not s:
        return "Sure, here is what you asked for"
    
    first, *rest_w = s.split(" ")
    rest = " ".join(rest_w)
    lead = {
        "write": "Sure, here is", 
        "create": "Sure, here is", 
        "generate": "Sure, here is",
        "compose": "Sure, here is", 
        "draft": "Sure, here is", 
        "design": "Sure, here is",
        "develop": "Sure, here is", 
        "give": "Sure, here is", 
        "provide": "Sure, here is",
        "make": "Sure, here is", 
        "produce": "Sure, here is", 
        "prepare": "Sure, here is",
        "list": "Sure, here are", 
        "explain": "Sure, here is an explanation of",
        "describe": "Sure, here is a description of", 
        "outline": "Sure, here is an outline of",
    }.get(first.lower())
    if lead is None:
        return "Sure, here is what you asked for"
    
    if rest.lower().startswith("me "):
        rest = rest[3:]
    rest = " ".join(rest.split()[:max_words])
    return f"{lead} {rest}".rstrip()


def load_sorry_bench_prompts(
    n_train: int,
    n_test: int,
    sample_seed: int = 0,
    dataset_name: str = "sorry-bench/sorry-bench-202406",
    split: str = "train",
    prompt_style: str = "base",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    
    import os
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name, 
        split=split, 
        token=os.getenv("HF_TOKEN") or None)
    
    # base only
    ds = ds.filter(lambda r: r["prompt_style"] == prompt_style)

    allrecs = [
        {
            "question_id": r["question_id"], 
            "category": r["category"],
            "instruction": "\n".join(r["turns"])
        }
        for r in ds
    ]
    rng = random.Random(sample_seed)
    pick = rng.sample(allrecs, n_train + n_test)
    print(f" >> SORRY-Bench {prompt_style}: {len(allrecs)} prompts; sampled {n_train} train + {n_test} test (seed={sample_seed})")
    return pick[:n_train], pick[n_train:]


def build_prompt(
    rec: dict[str, Any],
    tokenizer,
    embedding_layer,
    l_suf: int,
    device,
    target_max_words: int = 20,
) -> dict[str, Any]:
    '''
    Build the token scaffold for one prompt
    '''

    instruction = rec["instruction"]
    target = make_affirmative_target(instruction, max_words=target_max_words)

    messages = [{"role": "user", "content": instruction + " {optim}"}]

    templated = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
        
    before_str, after_str = templated.split("{optim}")


    def enc(s: str) -> torch.Tensor:
        return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

    before_ids, after_ids, target_ids = enc(before_str), enc(after_str), enc(target)

    with torch.no_grad():
        p: dict[str, Any] = {
            **rec, 
            "target": target,
            "before_ids": before_ids, 
            "after_ids": after_ids, 
            "target_ids": target_ids,
            "before_embeds": embedding_layer(before_ids),
            "after_embeds": embedding_layer(after_ids),
            "target_embeds": embedding_layer(target_ids),
            "n_before": len(before_ids), 
            "n_after": len(after_ids), 
            "n_target": len(target_ids),
        }
    p["target_start"] = p["n_before"] + l_suf + p["n_after"]
    return p


def build_prompts(
    raw_records: list[dict[str, Any]],
    tokenizer,
    embedding_layer,
    l_suf: int,
    device,
    target_max_words: int = 20,
) -> list[dict[str, Any]]:
    return [
        build_prompt(
            r, 
            tokenizer, 
            embedding_layer, 
            l_suf, 
            device,
            target_max_words=target_max_words
        )
        for r in raw_records
    ]


def build_disallowed_mask(tokenizer, vocab_size: int, device) -> torch.Tensor:
    '''
    Disallow the tokens that cannot be parsed and printed.
    '''
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(vocab_size):
        s = tokenizer.decode([i])
        if not (s.isascii() and s.isprintable()):
            mask[i] = True
    for t in tokenizer.all_special_ids:
        mask[t] = True
    return mask.to(device)


def init_suffix_ids(tokenizer, l_suf: int, device, init_token: str = "!") -> torch.Tensor:
    # Initial suffix: repeated '!'
    tok = tokenizer(init_token, add_special_tokens=False).input_ids[-1]
    return torch.full((l_suf,), tok, device=device, dtype=torch.long)
