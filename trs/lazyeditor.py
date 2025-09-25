#!/Data2/tangrui/miniconda3/envs/EasyEdit2/bin/python
import os
import sys
import json
import argparse

sys.path.append("..")
import easyeditor
from easyeditor import BaseEditor
import time
import math
import env


def ds_split(l: int, m: int, n: int):
    k = math.trunc(l / n)
    return k * (m - 1), l if m == n else k * m


def get_editor_args(args, pre_file):
    data_path = args.data_json
    ds_range = args.ds_range
    r_subject = []
    r_prompt = []
    r_target_new = []
    r_test = []
    edits = json.load(open(data_path))
    ds_r = [0, len(edits)]
    if ds_range != "All":
        if "q" in ds_range:
            ds_q = list(map(int, ds_range.split("q")))
            assert len(ds_q) == 2, "ds_range should be like '1q100'"
            ds_r = ds_split(len(edits), *ds_q)
        else:
            raise ValueError("ds_range should be like '1q100' or 'All'")
    edits = edits[ds_r[0] : ds_r[1]]
    print(f"Get {len(edits)} requests in range {ds_r[0]}:{ds_r[1]}.")
    for case in edits:
        r_subject.append(case["subject"])
        r_prompt.append(case["prompt"])
        r_target_new.append(case["target_new"])
        r_test.append(case["tests"])

    pre_edit = None
    if os.path.exists(pre_file):
        pre_edit = json.load(open(pre_file, "r"))
        print(f"Using pre edit cache: {pre_file}")
    else:
        print(f"No pre edit cache found: {pre_file}")

    return {
        "prompts": r_prompt,
        "target_new": r_target_new,
        "subject": r_subject,
        "portability_inputs": {
            "tests": {"prompt": r_test, "ground_truth": [""] * len(r_test)},
        },
        "pre_edit": pre_edit,
        "pre_file": pre_file,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str)
    parser.add_argument("--model_path", required=False, default=None, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--label", required=True, type=str)
    parser.add_argument("--data_json", required=True, type=str)
    parser.add_argument("--hparams_path", required=False, type=str, default=None)
    parser.add_argument("--ds_range", required=False, default="All", type=str)
    parser.add_argument("--rewrite_hparams", required=False, default=None, type=str)
    parser.add_argument("--outputs_dir", required=False, default="outputs", type=str)
    parser.add_argument("--prefiles_dir", required=False, default="prefiles", type=str)
    args = parser.parse_args()
    if args.ds_range == '1q1':
        args.ds_range = 'All'
    return args


def get_editor(args):
    editing_hparams = None
    algo = args.editing_method
    if args.editing_method in ("FT-L", "FT-M"):
        algo = "FT"
    for s in ("HyperParams", "Hparams"):
        try:
            editing_hparams = easyeditor.__getattribute__(algo + s)  # type: ignore
        except AttributeError:
            pass
    if editing_hparams is None:
        raise ValueError(f"Editing method {args.editing_method} not found.")
    if args.hparams_path is None:
        args.hparams_path = os.path.join("../hparams", algo, args.model_name)
    hparams = editing_hparams.from_hparams(args.hparams_path)

    if args.editing_method == "FT-L":
        hparams.objective_optimization = "prompt_last"
    elif args.editing_method == "FT-M":
        hparams.objective_optimization = "target_new"

    hparams.device = args.device
    if args.model_path:
        hparams.model_name = args.model_path
        hparams.tokenizer_name = args.model_path
    hparams.evaluation_type = "generate-text"
    if args.rewrite_hparams:
        d = eval(args.rewrite_hparams)
        for k, v in d.items():
            hparams.__setattr__(k, v)
    editor = BaseEditor.from_hparams(hparams)
    return editor


def run_edit(edit_args, editor):

    start_time = time.time()
    metrics, edited_model, _ = editor.edit(keep_original_weight=True, **edit_args)
    end_time = time.time()
    print(f"Time cost: {end_time-start_time:.2f}s")
    return metrics


def post_process(metrics):

    def metric_process(req):
        return {
            "pre": (
                req["pre"]["rewrite_gen_content"][0],
                req["pre"]["portability"]["tests_acc"],
            ),
            "post": (
                req["post"]["rewrite_gen_content"][0],
                req["post"]["portability"]["tests_acc"],
            ),
        }

    metrics = list(map(metric_process, metrics))

    return metrics


if __name__ == "__main__":
    args = parse_args()
    editor = get_editor(args)

    ds_name = env.get_ds_name(args.data_json)
    expenv = env.ExpEnv(
        args.model_name, ds_name, args.editing_method, args.label, args.ds_range, args.outputs_dir
    )

    pre_file = expenv.get_prefile_path(args.prefiles_dir)

    edit_args = get_editor_args(args, pre_file)

    metrics = run_edit(edit_args, editor)
    metrics = post_process(metrics)

    output_file = expenv.get_output_path()

    json.dump(metrics, open(output_file, "w"), indent=4)
