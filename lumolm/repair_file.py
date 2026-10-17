"""Real source-code edit: small single-function Python files only.

This uses neural edit outputs, NOT template codegen or reference target text.
It validates syntax but never executes produced code and never overwrites.
"""
import argparse
import ast
from pathlib import Path

from .repair import load, predict_edit
from .train import choose_device

SUPPORTED = {"fix":"FIX","refactor":"REFACTOR"}

def propose(model,source,kind):
    if kind not in SUPPORTED:
        raise ValueError("kind must be fix or refactor")
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError("supports one function only; large files must be split")
    func=tree.body[0]
    if len(func.args.args)!=1 or func.decorator_list or func.args.vararg or func.args.kwarg:
        raise ValueError("supports simple one-argument functions only")
    # The sample keeps its original function signature and source body.
    if any(not isinstance(n,(ast.Return,ast.Assign)) for n in func.body):
        raise ValueError("supports only simple assign/return function bodies")
    if len(func.body)>2:
        raise ValueError("supports at most two statements")
    text=source.strip()+"\n"
    prompt=SUPPORTED[kind]+"\n"+text+"EDIT:\n    "
    edit=predict_edit(model,prompt)
    if not edit.startswith("return "):
        raise ValueError("model did not output a return statement; no changes saved")
    if "\n" in edit or len(edit)>120:
        raise ValueError("model edit is not a single return statement")
    line=source.splitlines()[0]
    proposal=line+"\n    "+edit+"\n"
    ast.parse(proposal)
    # Reject dangerous generated call structures. Allow minimal known
    # functions only for this constrained proof-of-concept.
    tree=ast.parse(proposal)
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,
             ast.Name,ast.Load,ast.Constant,ast.BinOp,ast.UnaryOp,ast.Compare,
             ast.Add,ast.Sub,ast.Mult,ast.Mod,ast.Div,ast.FloorDiv,
             ast.Eq,ast.NotEq,ast.Gt,ast.Lt,ast.GtE,ast.LtE,
             ast.Call,ast.USub,ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node,allowed):
            raise ValueError("model emitted an unsupported code construct")
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Name) or node.func.id!="abs":
                raise ValueError("only abs() is allowed in this proof-of-concept")
    return proposal,edit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--kind",choices=tuple(SUPPORTED),required=True)
    p.add_argument("--device",choices=("cpu","cuda","auto","mps"),default="cpu")
    args=p.parse_args(argv)
    if not args.input.is_file():
        p.error("input file does not exist")
    if args.input.stat().st_size>8192:
        p.error("only tiny single-function inputs are supported")
    if args.output.exists() or args.output.is_symlink():
        p.error("output file already exists, refusing to overwrite")
    model=load(args.checkpoint,choose_device(args.device))
    before=args.input.read_text(encoding="utf8")
    proposed,edit=propose(model,before,args.kind)
    args.output.open("x",encoding="utf8").write(proposed)
    print("MODEL_EDIT="+repr(edit))
    print("CANDIDATE="+str(args.output)+" (syntax checked; NOT EXECUTED; review and test manually)")

if __name__=="__main__":
    main()
