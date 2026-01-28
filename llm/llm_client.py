#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
llm_client.py

LLM 客戶端封裝模組：
- 讀取 config.yaml 配置
- 使用 OpenAI SDK 調用 LLM
- 封裝 generate_docstring() 方法
- 包含重試邏輯和錯誤處理
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


@dataclass
class LLMStats:
    """LLM 調用統計"""
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: int = 0
    errors: list = field(default_factory=list)


def _find_config_yaml() -> str:
    """
    尋找 config.yaml，依序嘗試：
    1. 環境變數 LOCBENCH_CONFIG
    2. <locbench>/config.yaml（相對於此檔案往上兩層）
    3. 當前工作目錄的 config.yaml
    """
    if os.environ.get("LOCBENCH_CONFIG"):
        return os.environ["LOCBENCH_CONFIG"]

    # <locbench>/analyzer/llm/llm_client.py -> <locbench>/config.yaml
    here = os.path.dirname(os.path.abspath(__file__))
    locbench_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidate = os.path.join(locbench_root, "config.yaml")
    if os.path.isfile(candidate):
        return candidate

    if os.path.isfile("config.yaml"):
        return "config.yaml"

    raise FileNotFoundError(
        "找不到 config.yaml。請設置環境變數 LOCBENCH_CONFIG 或將 config.yaml 放在 locbench 根目錄。"
    )


def load_llm_config(config_path: Optional[str] = None) -> Dict:
    """
    讀取 config.yaml 中的 llm_chat 配置
    """
    if config_path is None:
        config_path = _find_config_yaml()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm_chat", {})
    return {
        "api_base_url": llm_cfg.get("api_base_url", ""),
        "api_key": llm_cfg.get("api_key", ""),
        "model_name": llm_cfg.get("model_name", "qwen3:235b"),
    }


class LLMClient:
    """
    LLM 客戶端封裝
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        retry_times: int = 2,
        retry_delay: float = 1.0,
        call_interval: float = 0.5,
    ):
        if OpenAI is None:
            raise ImportError("請先安裝 openai: pip install openai")

        self.config = load_llm_config(config_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.call_interval = call_interval
        self._last_call_time: float = 0.0

        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base_url"],
        )
        self.model_name = self.config["model_name"]
        self.stats = LLMStats()

    def _wait_for_rate_limit(self):
        """等待以避免觸發限流"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.call_interval:
            time.sleep(self.call_interval - elapsed)

    def _call_llm(self, messages: list, temperature: Optional[float] = None) -> Dict:
        """
        調用 LLM API（帶重試）
        返回: {"content": str, "prompt_tokens": int, "completion_tokens": int, "latency_ms": int}
        """
        self._wait_for_rate_limit()

        temp = temperature if temperature is not None else self.temperature
        last_error = None

        for attempt in range(self.retry_times + 1):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                latency_ms = int((time.time() - t0) * 1000)
                self._last_call_time = time.time()

                content = resp.choices[0].message.content or ""
                usage = resp.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                return {
                    "content": content,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                }

            except Exception as e:
                last_error = e
                if attempt < self.retry_times:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise last_error  # type: ignore

    def generate_docstring(
        self,
        file_source: str,
        func_qualname: str,
        func_lineno: int,
        style: str = "google",
        language: str = "English",
    ) -> Dict:
        """
        使用 LLM 生成 docstring

        Args:
            file_source: 整個檔案的源碼
            func_qualname: 函式的完整限定名（如 module.ClassName.method_name）
            func_lineno: 函式定義的行號
            style: docstring 風格 (google/numpy/sphinx)
            language: 輸出語言

        Returns:
            {
                "success": bool,
                "docstring": str,  # 生成的 docstring（不含三引號）
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": int,
                "error": Optional[str],
            }
        """
        style_desc = {
            "google": "Google style (Args/Returns/Raises sections)",
            "numpy": "NumPy style (Parameters/Returns sections)",
            "sphinx": "Sphinx reST style (:param/:returns: tags)",
        }.get(style.lower(), "Google style")

        # 構建 prompt
        system_prompt = f"""You are a Python docstring expert. Generate high-quality docstrings in {style_desc}.

Rules:
1. Output ONLY the docstring content (without triple quotes)
2. Include: brief description, Args/Parameters, Returns, Raises (if applicable)
3. Be concise but informative
4. Language: {language}
5. Do NOT include any explanation or markdown formatting outside the docstring"""

        user_prompt = f"""Generate a docstring for the following function.

## File Context
```python
{file_source}
```

## Target Function
- Qualified name: {func_qualname}
- Line number: {func_lineno}

Output the docstring content only (no triple quotes, no extra text):"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self.stats.total_calls += 1

        try:
            result = self._call_llm(messages)
            docstring = self._parse_docstring_response(result["content"])

            if not docstring.strip():
                self.stats.failed_calls += 1
                self.stats.errors.append(f"{func_qualname}: empty response")
                return {
                    "success": False,
                    "docstring": "",
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "latency_ms": result["latency_ms"],
                    "error": "empty_response",
                }

            self.stats.success_calls += 1
            self.stats.total_prompt_tokens += result["prompt_tokens"]
            self.stats.total_completion_tokens += result["completion_tokens"]
            self.stats.total_latency_ms += result["latency_ms"]

            return {
                "success": True,
                "docstring": docstring,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
                "error": None,
            }

        except Exception as e:
            self.stats.failed_calls += 1
            self.stats.errors.append(f"{func_qualname}: {str(e)}")
            return {
                "success": False,
                "docstring": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": str(e),
            }

    def _parse_docstring_response(self, content: str) -> str:
        """
        解析 LLM 回應，提取 docstring 內容
        處理可能的格式問題（如包含三引號、markdown 代碼塊等）
        """
        text = content.strip()

        # 移除 markdown 代碼塊
        if text.startswith("```"):
            lines = text.split("\n")
            # 找到第一個 ``` 後的內容
            start = 1
            end = len(lines)
            for i, line in enumerate(lines[1:], 1):
                if line.strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()

        # 移除三引號（如果 LLM 錯誤地包含了）
        if text.startswith('"""') and text.endswith('"""'):
            text = text[3:-3].strip()
        elif text.startswith("'''") and text.endswith("'''"):
            text = text[3:-3].strip()

        # 移除開頭的三引號（如果只有開頭）
        if text.startswith('"""'):
            text = text[3:].strip()
        if text.startswith("'''"):
            text = text[3:].strip()

        # 移除結尾的三引號
        if text.endswith('"""'):
            text = text[:-3].strip()
        if text.endswith("'''"):
            text = text[:-3].strip()

        # 移除 <think>...</think> 標籤（某些模型會輸出思考過程）
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        return text

    def get_stats_dict(self) -> Dict:
        """返回統計數據的字典形式"""
        return {
            "total_calls": self.stats.total_calls,
            "success_calls": self.stats.success_calls,
            "failed_calls": self.stats.failed_calls,
            "total_prompt_tokens": self.stats.total_prompt_tokens,
            "total_completion_tokens": self.stats.total_completion_tokens,
            "total_latency_ms": self.stats.total_latency_ms,
            "avg_latency_ms": (
                self.stats.total_latency_ms // self.stats.success_calls
                if self.stats.success_calls > 0
                else 0
            ),
            "errors": self.stats.errors[:10],  # 只保留前 10 個錯誤
        }


# 便捷函數
_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """獲取默認的 LLM 客戶端（單例）"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def generate_docstring(
    file_source: str,
    func_qualname: str,
    func_lineno: int,
    style: str = "google",
    language: str = "English",
) -> Dict:
    """
    便捷函數：使用默認客戶端生成 docstring
    """
    return get_default_client().generate_docstring(
        file_source=file_source,
        func_qualname=func_qualname,
        func_lineno=func_lineno,
        style=style,
        language=language,
    )


if __name__ == "__main__":
    # 簡單測試
    client = LLMClient()
    test_source = '''
def add(a: int, b: int) -> int:
    return a + b
'''
    result = client.generate_docstring(
        file_source=test_source,
        func_qualname="add",
        func_lineno=2,
        style="google",
    )
    print("Result:", result)
    print("Stats:", client.get_stats_dict())
