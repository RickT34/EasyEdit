from env import ExpEnv


def make_cmd_maker(cmd_addon_maker=None):
    def mk_exp_cmd(env: ExpEnv, env_file: str, device):
        return [
            "./lazyeditor.py",
            "--envfile",
            str(env_file),
            "--device",
            str(device),
        ] + (cmd_addon_maker(env, device) if cmd_addon_maker else [])

    return mk_exp_cmd
