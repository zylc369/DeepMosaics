# -*- coding: utf-8 -*-
# 环境/库初始化脚本

import sys, os, platform, subprocess, venv
from pathlib import Path
from packaging import version

# 获取当前脚本的完整路径
script_path = os.path.realpath(__file__)
# 获取脚本所在的目录
script_dir = os.path.dirname(script_path)
# 项目根目录
project_dir = Path(script_dir).parent
print("项目根目录：", project_dir)

other_dir = os.path.join(project_dir, 'util')
sys.path.append(os.path.abspath(other_dir))

try:
    import cmd_line_utils
except Exception as e:
    print(f'load lib failed: {e}')
    sys.exit(1)

pipCmd = 'pip'


# 检查虚拟环境是否已经创建，如果没有创建那么先创建
def check_and_create_python_venv() -> str:
    venv_name = 'dm'
    venv_dir = os.path.join(project_dir, 'venv', venv_name)

    if not os.path.exists(venv_dir):
        print(f"虚拟环境 {venv_name} 不存在，正在创建...")
        venv.create(venv_dir, with_pip=True)
        print(f"虚拟环境 {venv_name} 已成功创建。")
    else:
        print(f"虚拟环境 {venv_name} 已存在。")
    return venv_dir


# 安装Python依赖库
def install_python_libs(venv_dir: str):
    # 确定pip可执行文件的路径
    pip_executable = os.path.join(venv_dir, ('Scripts' if os.name == 'nt' else 'bin'), pipCmd)
    print("pip_executable=", pip_executable)

    print("安装Python依赖库：")
    cmd_line_utils.run([pip_executable, "install", "-r", os.path.join(project_dir, "requirements.txt")])

    print("")

    # https://pytorch.org/
    print("安装Python torch库(匹配CUDA 12.6计算架构)：")
    cmd_line_utils.run([pip_executable, "install", "-r", os.path.join(project_dir, "requirements_torch_cuda_12_6.txt")])


# 安装ffmpeg
def install_ffmpeg():
    min_version = "3.4.6"

    system = platform.system()

    if system == "Linux":
        # 假设使用的是基于Debian的发行版，例如Ubuntu
        print("检测到Linux系统，正在尝试安装ffmpeg...")
        subprocess.run(['sudo', 'apt-get', 'update'])
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'])
    elif system == "Darwin":  # macOS
        print("检测到macOS系统，正在尝试安装ffmpeg...")
        subprocess.run(['brew', 'install', 'ffmpeg'])
    elif system == "Windows":
        # 获得版本号
        package_name = "ffmpeg-full"
        installed_version = cmd_line_utils.get_installed_version_in_windows(package_name)
        print(f"检测到Windows系统，{package_name}的版本号是：{installed_version}")
        if installed_version is not None and version.parse(installed_version) >= version.parse(min_version):
            print(f"已安装的版本满足要求({min_version}及以上)，无需操作。")
        else:
            print(f"正在尝试安装{package_name}...")
            cmd_line_utils.run(["choco", "install", "-y", package_name])
    else:
        print(f"不支持的操作系统: {system}")
        exit(1)


# 初始化
def init():
    # 获得虚拟环境
    venv_dir = check_and_create_python_venv()
    print("虚拟环境路径：", venv_dir)
    print("")

    # 安装Python依赖库
    install_python_libs(venv_dir)

    # 安装ffmpeg
    install_ffmpeg()


if __name__ == '__main__':
    init()
