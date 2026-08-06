import argparse
import json
import os
import random
import re
import time
 
import numpy as np
 
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from datasets import load_dataset
 
DEFAULTS = dict(
    model          = "meta-llama/Llama-3.2-3B", #use Llama-3.1-8B for final run
    num_steps      = 1000,
    batch_size     = 8,
    lr             = 2e-4,
    lora_rank      = 32,
    max_seq_len    = 1024,
    checkpoint_name= "multitask-sft",
 
    #dataset mixing weights
    #maybe heavier on math & code since they improve faster

    #math
    mix_gsm8k = 0.35, 
    #code
    mix_code = 0.35, 
    #instruction following
    mix_ifeval = 0.30, 
 
    #dataset size caps per task (None = use all available)
    limit_per_task = None,
 
    #chain-of-thought + final answer
    gsm8k_system   = (
        "You are a helpful math tutor. Solve the problem step by step, "
        "then state your final answer as:\n#### <number>"
    ),
    #bare Python function
    #model can see a system prompt during training but strip fences at eval time via the renderer
    code_system    = (
        "You are an expert Python programmer. "
        "Complete the function below. "
        "Return only valid Python code with no markdown fences."
    ),
    ifeval_system  = (
        "You are a helpful, precise assistant. "
        "Follow the user's instructions exactly, including any formatting constraints."
    ),
)

#load data
def load_gsm8k_data(limit=None):  
    ds = load_dataset("openai/gsm8k", "main", split="train") #TRAINING ONLY!
    
    if limit: #only for smaller training, not the actual
        ds = ds.select(range(min(limit, len(ds))))
 
    conversations = []
    for row in ds:
        question = row["question"].strip()
        answer = row["answer"].strip() 
        convo = [
            {"role": "system",    "content": DEFAULTS["gsm8k_system"]},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        conversations.append(convo)
 
    #DEBUG 
    print(f"{len(conversations)} GSM8K examples")
    return conversations
 
 
def load_code_data(limit=None):
   #'instruction' and 'response' columns
    #limit response length.
 
    """Try filtering to only HumanEval-style function completion problems 
    or add unit-test solutions from other datasets (e.g. bigcode/the-stack).
    """

    effective_limit = limit or 30_000 #in case limit is None, cap 
    ds = load_dataset(
        "nvidia/OpenCodeInstruct",
        split="train", #train split only 
        streaming=True,
    )
 
    conversations = []
    for row in ds:
        if len(conversations) >= effective_limit:
            break
 
        instruction = (row.get("input") or "").strip()
        response = (row.get("output") or "").strip()
 
        if not instruction or not response:
            continue
 
        #added a quality filter that skips very short responses
        if len(response) < 30:
            continue
 
        convo = [
            {"role": "system", "content": DEFAULTS["code_system"]},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        conversations.append(convo)
 
    #DEBUG
    print(f"{len(conversations)} code examples")

    return conversations

def load_orca_math_data(limit=None):
    print("  Loading Orca Math data...")
    effective_limit = limit or 20_000
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    if limit:
        ds = ds.select(range(min(effective_limit, len(ds))))

    conversations = []
    for row in ds:
        question = (row.get("question") or "").strip()
        answer   = (row.get("answer")   or "").strip()
        if not question or not answer:
            continue
        convo = [
            {"role": "system",    "content": DEFAULTS["gsm8k_system"]},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        conversations.append(convo)

    print(f"    → {len(conversations)} orca math examples")
    return conversations
 
 
def load_ifeval_data(limit=None):
    """there are 'ifeval' or 'wildchat' subsets, which more directly resemble the eval distribution"""
    #print("Loading IFEval data (allenai/tulu-3-sft-mixture, train split)…")
 
    effective_limit = limit or 20_000
    ds = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
        streaming=True,
    )
 
    conversations = []

    #after dev 2, found that IDEval data is very broad 
    #EXTENSION 1!!!! Quality Filtering on the data 
    GOOD_SOURCES = {"wildchat", "ifeval", "open_assistant", "sharegpt"}

    for row in ds:
        if len(conversations) >= effective_limit:
            break
 
        messages = row.get("messages", [])
        if not messages:
            continue

        source = row.get("source", "")
        if source and not any(s in source.lower() for s in GOOD_SOURCES):
            continue
        
        #normalise to list-of-dicts with role/content keys
        convo = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role and content:
                convo.append({"role": role, "content": content})
 
        #must have at least one user + assistant turn
        roles = [m["role"] for m in convo]
        if "user" not in roles or "assistant" not in roles:
            continue
 
        #prepend system prompt if none exists
        if convo[0]["role"] != "system":
            convo.insert(0, {"role": "system", "content": DEFAULTS["ifeval_system"]})
 
        conversations.append(convo)
 
    #DEBUG 
    print(f"{len(conversations)} instruction-following examples")

    return conversations
 
 

#after loading the data, create a miz of the datasets 
 
def build_mixed_dataset(gsm8k_convos, code_convos, ifeval_convos, mix_weights):
    """ Build a single flat list of conversations that respects the target mixing ratio; 
    Strategy: weighted random sampling with replacement until we have enough
    data to fill num_steps * batch_size examples (before cycling). """
    
    sources = [
        ("gsm8k",  gsm8k_convos,  mix_weights["gsm8k"]),
        ("code",   code_convos,   mix_weights["code"]),
        ("ifeval", ifeval_convos, mix_weights["ifeval"]),
    ]
 
    total = sum(len(s[1]) for s in sources)
    target_n = max(total, 2000)   # at least 2000 unique examples before cycling
 
    mixed = []
    weights_norm = [w for _, _, w in sources]
    w_sum = sum(weights_norm)
    weights_norm = [w / w_sum for w in weights_norm]
 
    rng = random.Random(42)
    for _ in range(target_n):
        task_idx = rng.choices(range(len(sources)), weights=weights_norm, k=1)[0]
        _, pool, _ = sources[task_idx]
        mixed.append(rng.choice(pool))
 
    rng.shuffle(mixed)
    print(f"Mixed dataset: {len(mixed)} examples "
          f"(GSM8K {mix_weights['gsm8k']:.0%} / "
          f"Code {mix_weights['code']:.0%} / "
          f"IF {mix_weights['ifeval']:.0%})")
    
    return mixed
 
def convert_to_datums(conversations, renderer, max_length):
    """chat conversations -> tinker training datums"""
    datums = []
    skipped = 0
    for convo in conversations:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            datums.append(datum)
        except Exception:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} examples (too long or malformed)")
    return datums
 
#training 
def train(args):
    print("\n")
    print(f"Model: {args.model}")
    print(f"Steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Mix weights:  GSM8K={args.mix_gsm8k:.2f}  Code={args.mix_code:.2f}  IF={args.mix_ifeval:.2f}")
 
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    #print(f"Renderer: {renderer_name}")
 
    #DEBUG
    print("\n[1/4] Loading datasets…")
    limit = args.limit_per_task
 
    gsm8k_convos = load_gsm8k_data(limit)
    
    #NEW
    #orca_convos   = load_orca_math_data(limit)
    #gsm8k_convos  = gsm8k_convos + orca_convos 
    #NEWWWW

    code_convos = load_code_data(limit)
    ifeval_convos = load_ifeval_data(limit)

    #DEBUG 
    print(f"GSM8K: {len(gsm8k_convos)} | Code: {len(code_convos)} | IFEval: {len(ifeval_convos)}")
 
    mix_weights = dict(
        gsm8k = args.mix_gsm8k,
        code = args.mix_code,
        ifeval = args.mix_ifeval,
    )

    mixed_convos = build_mixed_dataset(gsm8k_convos, code_convos, ifeval_convos, mix_weights)
    
    #DEBUG 
    print("\n[2/4] Tokenising…")
    all_datums = convert_to_datums(mixed_convos, renderer, args.max_seq_len)
    ifeval_datums = convert_to_datums(ifeval_convos, renderer, args.max_seq_len)  #EXTENSIONNNNN 2 
    #DEBUG 
    #print(f"Total datums ready: {len(all_datums)}")
 
    if len(all_datums) == 0:
        raise RuntimeError("no data so check dataset loading")
 
    #DEBUG 
    print("\n[3/4] Creating LoRA training client…")
    sc = tinker.ServiceClient()

    tc = sc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    #print(f"Training client ready (rank={args.lora_rank})")

    #DEBUG
    print(f"\n[4/4] Training for {args.num_steps} steps…")
    adam_params = types.AdamParams(
        learning_rate = args.lr,
        beta1         = 0.9,
        beta2         = 0.95,
        eps           = 1e-8,
    )
 
    losses = []
    save_paths = [] #intermediate 
    n = len(all_datums)
    t0 = time.time()
 
    for step in range(args.num_steps):
        #cycle through the dataset deterministically.  #TEST THAT FAILEDDDD
        #for step in range(args.num_steps):
        
        #-------------------------------------------------
        # Curriculum: first 25% of steps use IFEval only,
        # then gradually introduce math and code
        '''
        progress = step / args.num_steps
        if progress < 0.25:
            # Phase 1: IFEval only
            pool = ifeval_datums
        else:
            # Phase 2: full mix
            pool = all_datums

        start = (step * args.batch_size) % len(pool)
        batch = [pool[(start + i) % len(pool)] for i in range(args.batch_size)]'''
        #-------------------------------------------------

        start = (step * args.batch_size) % n
        batch = [all_datums[(start + i) % n] for i in range(args.batch_size)]
        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future   = tc.optim_step(adam_params)
 
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()
 
        #compute token-weighted loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        losses.append(float(loss))
 
        if (step + 1) % 50 == 0 or step == 0:
            elapsed  = time.time() - t0
            avg_loss = np.mean(losses[-50:])
            print(f"Step {step+1:>5}/{args.num_steps} | "
                  f"Loss: {loss:.4f} | "
                  f"Avg(50): {avg_loss:.4f} | "
                  f"Elapsed: {elapsed:.0f}s")
 
        #save a checkpoint every 500 steps
        if (step + 1) % 500 == 0 and (step + 1) < args.num_steps:
            ckpt_name = f"{args.checkpoint_name}-step{step+1}"
            print(f"  Saving intermediate checkpoint '{ckpt_name}'…")
            ckpt = tc.save_weights_for_sampler(name=ckpt_name).result()
            save_paths.append((step + 1, ckpt.path))
            print(f"    → {ckpt.path}")
 
    #save the last checkpoint 
    print(f"\nSaving final checkpoint '{args.checkpoint_name}'…")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    save_paths.append((args.num_steps, checkpoint_path))
    print(f"  → {checkpoint_path}")
 
    #publish the last checkpoint
    if not args.no_publish:
        print("\nPublishing checkpoint…")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("Published!")
 
        #publish intermediate checkpoints as well 
        for step_n, path in save_paths[:-1]:
            try:
                rest_client.publish_checkpoint_from_tinker_path(path).result()
                print(f"Published intermediate checkpoint (step {step_n}): {path}")
            except Exception as e:
                print(f"Warning: could not publish intermediate checkpoint: {e}")
 
    info = {
        "checkpoint_path": checkpoint_path,
        "intermediate_checkpoints": [{"step": s, "path": p} for s, p in save_paths[:-1]],
        "base_model": args.model,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.lora_rank,
            "max_seq_len": args.max_seq_len,
            "mix_gsm8k": args.mix_gsm8k,
            "mix_code": args.mix_code,
            "mix_ifeval": args.mix_ifeval,
            "limit_per_task": args.limit_per_task,
            "total_datums": len(all_datums),
        },
        "losses": losses,
        "published": not args.no_publish,
    }
 
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")

    #final instructions 
    print("DONE:\n")
    print(f'python evaluation/eval_all.py \\')
    print(f'--checkpoint_path "{checkpoint_path}" \\')
    print(f'--base_model {args.model}')
    if save_paths[:-1]:
        print(f"\nAlso evaluate intermediate checkpoints to find the best:")
        for step_n, path in save_paths[:-1]:
            print(f'# step {step_n}: python evaluation/eval_all.py --checkpoint_path "{path}" --base_model {args.model} --limit 50')
    print()
 
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-task SFT training: IFEval + GSM8K + HumanEval"
    )
    p.add_argument("--model", type=str,   default=DEFAULTS["model"])
    p.add_argument("--num_steps", type=int,   default=DEFAULTS["num_steps"])
    p.add_argument("--batch_size", type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--lora_rank", type=int,   default=DEFAULTS["lora_rank"])
    p.add_argument("--checkpoint_name", type=str,   default=DEFAULTS["checkpoint_name"])
    p.add_argument("--max_seq_len", type=int,   default=DEFAULTS["max_seq_len"])
 
    #mixing weights
    p.add_argument("--mix_gsm8k", type=float, default=DEFAULTS["mix_gsm8k"],
                   help="Fraction of batch drawn from GSM8K data")
    p.add_argument("--mix_code", type=float, default=DEFAULTS["mix_code"],
                   help="Fraction of batch drawn from code data")
    p.add_argument("--mix_ifeval", type=float, default=DEFAULTS["mix_ifeval"],
                   help="Fraction of batch drawn from instruction-following data")
    #dataset size limits
    p.add_argument("--limit_per_task", type=int, default=DEFAULTS["limit_per_task"],
                   help="Cap examples per task (None=all). Use ~500 for smoke tests.")
    p.add_argument("--no_publish", action="store_true",
                   help="Skip publishing checkpoint to Tinker")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
 
    #validate mixing weights
    total_weight = args.mix_gsm8k + args.mix_code + args.mix_ifeval
    if abs(total_weight - 1.0) > 1e-3:
        print(f"Warning: mix weights sum to {total_weight:.3f}, normalising to 1.0")
        args.mix_gsm8k  /= total_weight
        args.mix_code   /= total_weight
        args.mix_ifeval /= total_weight
 
    train(args)
