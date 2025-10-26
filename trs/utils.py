from env import ExpEnv


def make_cmd_maker(cmd_addon_maker=None):
    def mk_exp_cmd(env: ExpEnv, device):
        return [
            "./lazyeditor.py",
            "--envjson",
            env.to_json(),
            "--device",
            str(device),
        ] + (cmd_addon_maker(env, device) if cmd_addon_maker else [])

    return mk_exp_cmd
