"""Task-specific neural edit learner. NO symbolic patches at inference.

Pairs are tiny Python buggy/refactor-before functions -> correct return lines.
Inference outputs raw neural bytes only. Cannot repair arbitrary applications.
"""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from .model import GPT, GPTConfig
from .train import choose_device

TASK="neural_python_short_edits_v1"
SEED=20261017
VARS=("x","n","v","value","number","amount","score","count","item","num","data","arg","input","k","p","q")
# [name, task, broken function body, expected new return line]
PATTERNS=(
 ("fix_even","fix","return {v} % 2 == 1","return {v} % 2 == 0"),
 ("fix_odd","fix","return {v} % 2 == 0","return {v} % 2 != 0"),
 ("fix_square","fix","return {v} + {v}","return {v} * {v}"),
 ("fix_increment","fix","return {v} - 1","return {v} + 1"),
 ("fix_positive","fix","return {v} < 0","return {v} > 0"),
 ("fix_absolute","fix","return {v}","return abs({v})"),
 ("refactor_square","refactor","temp = {v} * {v}\n    return temp","return {v} * {v}"),
 ("refactor_increment","refactor","result = {v} + 1\n    return result","return {v} + 1"),
 ("refactor_even","refactor","flag = {v} % 2 == 0\n    return flag","return {v} % 2 == 0"),
 ("refactor_absolute","refactor","answer = abs({v})\n    return answer","return abs({v})"),
)

def record(pattern,var,serial):
    name,kind,broken,fixed=pattern
    func=f"{name}_{serial}"
    body=broken.format(v=var)
    target=fixed.format(v=var)
    prefix=f"{kind.upper()}\ndef {func}({var}):\n    {body}\nEDIT:\n    "
    return {"pattern":name, "kind":kind,"var":var,"name":func,
            "prompt":prefix,"answer":target+"\n",
            "broken":f"def {func}({var}):\n    {body}\n",
            "correct":f"def {func}({var}):\n    {target}\n"}

def build_records():
    train,test=[],[]
    for i,pat in enumerate(PATTERNS):
        for j,var in enumerate(VARS):
            for ident in range(48):
                example=record(pat,var,ident)
                # Disjoint full code pair combinations, NOT disjoint algorithms:
                split=(ident*17+j*11+i*7)%11==0
                (test if split else train).append(example)
    rng=random.Random(SEED)
    rng.shuffle(train);rng.shuffle(test)
    return train,test

def masked_batch(records,count,block_size,device):
    rows=random.choices(records,k=count)
    x=torch.zeros((count,block_size),dtype=torch.long)
    y=torch.full((count,block_size),-100,dtype=torch.long)
    for i,row in enumerate(rows):
        prefix=row["prompt"].encode("utf8")
        answer=row["answer"].encode("utf8")
        full=prefix+answer
        if len(full)>block_size+1:
            raise ValueError(f"record too long: {row['name']}")
        k=len(full)-1
        x[i,:k]=torch.tensor(list(full[:-1]))
        y[i,len(prefix)-1:k]=torch.tensor(list(answer))
    return x.to(device),y.to(device)

@torch.no_grad()
def predict_edit(model,prompt):
    raw=prompt.encode("utf8")
    if len(raw)>=model.config.block_size:
        raise ValueError("repair input exceeds byte-level context")
    output=model.generate(raw,60,temperature=0,stop_sequences=(b"\n",))
    return output[len(raw):].decode("utf8",errors="replace").strip()

@torch.no_grad()
def evaluate(model,rows,limit=0):
    tests=rows[:limit] if limit else rows
    correct=0; by_pattern={};samples=[]
    for row in tests:
        result=predict_edit(model,row["prompt"])
        valid=result==row["answer"].strip()
        correct+=valid
        if row["pattern"] not in by_pattern: by_pattern[row["pattern"]]=[0,0]
        by_pattern[row["pattern"]][0]+=valid
        by_pattern[row["pattern"]][1]+=1
        if len(samples)<12:
            samples.append({"task":row["pattern"],"model":result,
                            "expected":row["answer"].strip(),"passed":valid})
    report={"correct":correct,"total":len(tests),
            "accuracy":round(correct/len(tests),6) if tests else None,
            "per_pattern":by_pattern,"examples":samples}
    print("REPAIR_EVAL="+json.dumps(report,ensure_ascii=True),flush=True)
    return report

def load(path,device="cpu"):
    state=torch.load(path,map_location="cpu",weights_only=True)
    if state.get("task")!=TASK: raise ValueError("not a short-code-repair checkpoint")
    model=GPT(GPTConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()

def train(args):
    torch.manual_seed(SEED);random.seed(SEED);torch.set_num_threads(args.threads)
    device=choose_device(args.device)
    training,heldout=build_records()
    cfg=GPTConfig(block_size=192,n_layer=3,n_head=4,n_embd=96)
    model=GPT(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    print(f"REPAIR_TRAIN examples={len(training)} heldout={len(heldout)} "
          f"parameters={sum(p.numel() for p in model.parameters())} steps={args.steps}",flush=True)
    best=float("inf")
    for step in range(1,args.steps+1):
        model.train()
        x,y=masked_batch(training,args.batch_size,cfg.block_size,device)
        optimizer.zero_grad(set_to_none=True)
        _,loss=model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step%args.eval_interval==0 or step==args.steps:
            with torch.no_grad():
                model.eval()
                vx,vy=masked_batch(heldout,48,cfg.block_size,device)
                _,val_loss=model(vx,vy)
            value=float(val_loss.item())
            print(f"REPAIR_STEP={step} loss={loss.item():.4f} heldout_token_loss={value:.4f}",flush=True)
            if value<best:
                best=value
                args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
                torch.save({"format_version":1,"config":asdict(cfg),
                            "model":model.state_dict(),"task":TASK,
                            "step":step,"best_val_loss":value,
                            "train_examples":len(training),"test_examples":len(heldout)},
                           args.checkpoint)
    best_model=load(args.checkpoint,device)
    evaluate(best_model,heldout,args.eval_limit)
    for ex in (
        "FIX\ndef is_even(n):\n    return n % 2 == 1\nEDIT:\n    ",
        "REFACTOR\ndef square(x):\n    result = x * x\n    return result\nEDIT:\n    ",
    ):
        print("REPAIR_PROBE="+json.dumps(
            {"prompt":ex,"model_output":predict_edit(best_model,ex)}),flush=True)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train",action="store_true")
    modes.add_argument("--evaluate",action="store_true")
    modes.add_argument("--prompt",help="literal compact edit prompt")
    ap.add_argument("--checkpoint",type=Path,default=Path("checkpoints/repair.pt"))
    ap.add_argument("--steps",type=int,default=8000)
    ap.add_argument("--eval-interval",type=int,default=1000)
    ap.add_argument("--batch-size",type=int,default=48)
    ap.add_argument("--eval-limit",type=int,default=250)
    ap.add_argument("--lr",type=float,default=7e-4)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--device",choices=("auto","cpu","cuda","mps"),default="cpu")
    args=ap.parse_args(argv)
    if min(args.steps,args.eval_interval,args.batch_size,args.threads)<1: ap.error("positive params required")
    if args.train:train(args)
    else:
        torch.set_num_threads(args.threads)
        model=load(args.checkpoint,choose_device(args.device))
        if args.evaluate:
            _,rows=build_records();evaluate(model,rows,args.eval_limit)
        else:print("MODEL_EDIT="+repr(predict_edit(model,args.prompt)))

if __name__=="__main__": main()
