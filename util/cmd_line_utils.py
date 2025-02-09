# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any
import os
import subprocess


def args2cmd(args: List[str]):
    cmd = ''
    for arg in args:
        cmd += (arg + ' ')
    return cmd


def run(args: List[str], mode: int = 0) -> list[bytes]:
    if mode == 0:
        cmd = args2cmd(args)
        os.system(cmd)

    elif mode == 1:
        '''
        out_string = os.popen(cmd_str).read()
        For chinese path in Windows
        https://blog.csdn.net/weixin_43903378/article/details/91979025
        '''
        cmd = args2cmd(args)
        stream = os.popen(cmd)._stream
        sout = stream.buffer.read().decode(encoding='utf-8')
        return sout

    elif mode == 2:
        cmd = args2cmd(args)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sout = p.stdout.readlines()
        return sout


def get_installed_version_in_windows(package_name: str) -> Optional[str]:
    """通过Chocolatey查询安装状态和版本"""
    try:
        # 使用choco list来查找所有已安装的包，-e 是严格比较包名
        result = subprocess.run(['choco', 'list', '-e', package_name],
                                capture_output=True, text=True, check=True)

        # 解析输出以找到ffmpeg相关的行
        for line in result.stdout.splitlines():
            if package_name in line.lower():  # 忽略大小写匹配
                parts = line.split()
                if len(parts) > 1:  # 确保有版本信息
                    package_name = parts[0].lower()  # 转换为小写进行比较
                    if package_name == package_name:
                        version = parts[1]
                        return version
        return None
    except subprocess.CalledProcessError as e:
        print(f"执行choco命令出错: {e}")
        return None
