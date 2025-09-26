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


def get_editor_args(dataset: env.DatasetEnv, pre_file):
    data_path = dataset.path
    ds_range = dataset.range
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


def parse_env():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envfile", required=True, type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--prefiles_dir", required=False, default="prefiles", type=str)
    args = parser.parse_args()
    expenv = env.loadEnv(args.envfile)
    expenv.tags.update(args.__dict__)
    return expenv


def get_editor(expenv: env.ExpEnv):
    editing_hparams = None
    algo = expenv.algo.name
    for s in ("HyperParams", "Hparams"):
        try:
            editing_hparams = getattr(easyeditor, algo + s)
        except AttributeError:
            pass
    if editing_hparams is None:
        raise ValueError(f"Editing method {algo} not found.")
    hparams = editing_hparams.from_hparams(
        os.path.join(expenv.algo.hparams_dir, expenv.model.name)
    )

    hparams.device = expenv.tags["device"]
    hparams.model_name = expenv.model.path
    hparams.tokenizer_name = expenv.model.path
    hparams.evaluation_type = "generate-text"
    for k, v in expenv.algo.hparams_rewrite.items():
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
            t: (
                req[t]["rewrite_gen_content"][0],
                req[t]["portability"]["tests_acc"],
            )
            for t in ("pre", "post")
        }

    metrics = list(map(metric_process, metrics))

    return metrics


def main():
    expenv = parse_env()

    editor = get_editor(expenv)

    pre_file = expenv.get_prefile_path(expenv.tags["prefiles_dir"])

    edit_args = get_editor_args(expenv.dataset, pre_file)

    metrics = run_edit(edit_args, editor)
    metrics = post_process(metrics)

    output_file = expenv.get_output_path()

    json.dump(metrics, open(output_file, "w"), indent=4)

if __name__ == "__main__":
    main()