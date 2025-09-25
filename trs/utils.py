from env import ExpEnv


def make_cmd_maker(cmd_addon_maker=None):
    def mk_exp_cmd(env: ExpEnv, device):
        return [
            "./lazyeditor.py",
            "--editing_method",
            env.algo,
            "--model_name",
            env.model_name,
            "--device",
            str(device),
            "--ds_range",
            env.ds_range,
            "--label",
            env.label,
            "--model_path",
            env.tags["model_path"],
            "--outputs_dir",
            env.outputs_dir,
            "--data_json",
            env.tags["data_json"],
        ] + (cmd_addon_maker(env, device) if cmd_addon_maker else [])

    return mk_exp_cmd
