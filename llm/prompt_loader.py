#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prompt_loader.py

統一的 prompt 模板載入模組。
所有 prompt 模板存放在 llm/prompts/ 目錄下，使用 Python str.format() 佔位符。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

# prompts 目錄路徑
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


@lru_cache(maxsize=32)
def _read_template(name: str) -> str:
    """
    讀取 prompts/ 目錄下的模板文件（帶快取）。

    Args:
        name: 模板文件名（如 "docstring_system.txt"）

    Returns:
        模板內容字串

    Raises:
        FileNotFoundError: 找不到指定的模板文件
    """
    path = os.path.join(_PROMPTS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt 模板文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_prompt(name: str, **kwargs) -> str:
    """
    載入並渲染 prompt 模板。

    Args:
        name: 模板文件名（如 "docstring_system.txt"）
        **kwargs: 用於填充模板佔位符的參數

    Returns:
        渲染後的 prompt 字串

    Examples:
        >>> load_prompt("docstring_system.txt", style_desc="Google style", language="English")
        'You are a Python docstring expert...'
    """
    template = _read_template(name)
    if kwargs:
        return template.format(**kwargs)
    return template


def load_prompt_pair(prefix: str, **kwargs) -> Dict[str, str]:
    """
    載入一組 system + user prompt。

    Args:
        prefix: 模板前綴（如 "docstring" 會載入 docstring_system.txt 和 docstring_user.txt）
        **kwargs: 用於填充模板佔位符的參數。
                  system_ 前綴的參數只傳給 system prompt，
                  user_ 前綴的參數只傳給 user prompt，
                  其餘參數兩者共享。

    Returns:
        {"system": str, "user": str}
    """
    shared = {}
    system_only = {}
    user_only = {}

    for k, v in kwargs.items():
        if k.startswith("system_"):
            system_only[k[7:]] = v  # 去掉 system_ 前綴
        elif k.startswith("user_"):
            user_only[k[5:]] = v    # 去掉 user_ 前綴
        else:
            shared[k] = v

    system_kwargs = {**shared, **system_only}
    user_kwargs = {**shared, **user_only}

    return {
        "system": load_prompt(f"{prefix}_system.txt", **system_kwargs),
        "user": load_prompt(f"{prefix}_user.txt", **user_kwargs),
    }


def list_prompts() -> list:
    """列出所有可用的 prompt 模板文件"""
    if not os.path.isdir(_PROMPTS_DIR):
        return []
    return sorted(f for f in os.listdir(_PROMPTS_DIR) if f.endswith(".txt"))
