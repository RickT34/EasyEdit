import subprocess
import time
from multiprocessing import Process
from pathlib import Path
import mail
from env import ExpEnv, dumpEnvs


def run_cmd(command: list[str], output_file: Path):
    """运行单个实验并将输出重定向到文件"""
    try:
        # 输出文件路径处理
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)

        # 打印开始信息
        print(f"    开始实验: {command}")
        print(f"    输出文件: {output_file}")
        start_time = time.time()

        # 执行命令并捕获输出
        with open(output_file, "w") as f:
            process = subprocess.Popen(
                command,
                stdout=f,
                stderr=f,
            )
            process.wait()  # 等待进程完成

            # 计算运行时间
            duration = time.time() - start_time
            msg = f"    实验完成: 耗时 {duration:.2f} 秒. 状态码: {process.returncode}"
            print(msg)
            f.write(f"\n\n=== {msg} ===\n")
            f.write(f"实验命令: {' '.join(command)}\n")

    except Exception as e:
        print(f"实验错误: {command} : {e}")

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
def run_exps(name:str, envs: list[ExpEnv], devices:list[int], cmd_maker):
    env_files = dumpEnvs(envs, name)
    running:dict[int, Process] = {}
    for env, env_file in zip(envs, env_files):
        if env.get_output_path().exists():
            print(f"Skipping {env.label} because output file exists.")
            continue
        d = None
        while d is None:
            for i in devices:
                if i not in running or not running[i].is_alive():
                    d = i
                    break
            else:
                time.sleep(10)
        cmd = cmd_maker(env, env_file, d)
        logdir = env.get_output_path().parent
        log_file = logdir / f"exp_{get_timestamp()}.log"
        p = Process(target=run_cmd, args=(cmd, log_file))
        p.start()
        running[d]=p
    for p in running.values():
        p.join()
    try:
        mail.send_default("ICT-v2", f"Exp Done: {envs}")
    except:
        print("Failed to send email.")