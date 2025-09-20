import subprocess
import time
from multiprocessing import Process
from pathlib import Path
import mail


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

    except Exception as e:
        print(f"实验错误: {command} : {e}")


def run_cmds(cmd_output_pairs: list[tuple[list[str], Path]]):
    print("开始运行实验组...")
    processes = []
    for command, output_file in cmd_output_pairs:
        p = Process(target=run_cmd, args=(command, output_file))
        p.start()
        processes.append(p)
    # 等待所有进程完成
    for p in processes:
        p.join()
    print("实验组运行完成.")

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def run_exp(log_dir:str, algo: str, label: str, devices:list[int], cmd_maker):
    try:
        logdir = Path(log_dir) / label / algo
        logdir.mkdir(parents=True, exist_ok=True)
        timestamp = get_timestamp()
        cmd_out = []
        for i, device in enumerate(devices):
            cmd_out.append(
                (
                    cmd_maker(i),
                    logdir / f"exp_{timestamp}_{i+1}.log",
                )
            )
        run_cmds(cmd_out)
        try:
            mail.send_default("ICT-v2", f"Exp Done: {algo} {label}")
        except:
            print("Failed to send email.")
    except Exception as e:
        print(f"Exp Error: {e}")