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
import json as _json
from typing import Dict, List, Optional

import yaml

from .prompt_loader import load_prompt

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
    # d:\locbench\config.yaml
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

        # 從外部模板載入 prompt
        system_prompt = load_prompt(
            "docstring_system.txt",
            style_desc=style_desc,
            language=language,
        )
        user_prompt = load_prompt(
            "docstring_user.txt",
            file_source=file_source,
            func_qualname=func_qualname,
            func_lineno=func_lineno,
        )

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

    def _parse_structured_response(self, content: str) -> Dict[str, str]:
        """
        解析包含 [DOCSTRING] 和 [ANALYSIS] 標籤的結構化回應。

        Returns:
            {"docstring": str, "analysis": str}
            如果解析失敗，docstring 回退到整段文本，analysis 為空。
        """
        text = content.strip()
        # 移除 <think>...</think> 標籤
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        docstring = ""
        analysis = ""

        # 提取 [DOCSTRING]...[/DOCSTRING]
        m_doc = re.search(r"\[DOCSTRING\]\s*(.*?)\s*\[/DOCSTRING\]", text, re.DOTALL)
        if m_doc:
            docstring = m_doc.group(1).strip()

        # 提取 [ANALYSIS]...[/ANALYSIS]
        m_ana = re.search(r"\[ANALYSIS\]\s*(.*?)\s*\[/ANALYSIS\]", text, re.DOTALL)
        if m_ana:
            analysis = m_ana.group(1).strip()

        # 如果結構化解析失敗，回退：把整段文本當 docstring
        if not docstring:
            docstring = self._parse_docstring_response(text)

        # 清理 docstring 中的三引號
        docstring = self._clean_docstring_quotes(docstring)

        return {"docstring": docstring, "analysis": analysis}

    def _clean_docstring_quotes(self, text: str) -> str:
        """移除 docstring 中可能殘留的三引號和 markdown 代碼塊。"""
        # 移除 markdown 代碼塊
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            end = len(lines)
            for i, line in enumerate(lines[1:], 1):
                if line.strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()

        if text.startswith('"""') and text.endswith('"""'):
            text = text[3:-3].strip()
        elif text.startswith("'''") and text.endswith("'''"):
            text = text[3:-3].strip()
        if text.startswith('"""'):
            text = text[3:].strip()
        if text.startswith("'''"):
            text = text[3:].strip()
        if text.endswith('"""'):
            text = text[:-3].strip()
        if text.endswith("'''"):
            text = text[:-3].strip()
        return text

    def _get_style_desc(self, style: str) -> str:
        """將風格名稱轉換為描述文字。"""
        return {
            "google": "Google style (Args/Returns/Raises sections)",
            "numpy": "NumPy style (Parameters/Returns sections)",
            "sphinx": "Sphinx reST style (:param/:returns: tags)",
        }.get(style.lower(), "Google style")

    def generate_docstring_with_diff(
        self,
        old_code: str,
        new_code: str,
        func_qualname: str,
        style: str = "google",
        language: str = "English",
        commit_hash: str = "",
    ) -> Dict:
        """
        Hop 0 專用：基於 diff（舊代碼 + 新代碼）生成 docstring 並產出變更分析。

        Args:
            old_code: 修改前的函式源碼（可能為空字串表示新增函式）
            new_code: 修改後的函式源碼
            func_qualname: 函式完整限定名
            style: docstring 風格
            language: 輸出語言
            commit_hash: commit SHA（用於在 docstring 的 Changelog 段落中引用）

        Returns:
            {
                "success": bool,
                "docstring": str,
                "change_analysis": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": int,
                "error": Optional[str],
            }
        """
        style_desc = self._get_style_desc(style)

        system_prompt = load_prompt(
            "docstring_diff_system.txt",
            style_desc=style_desc,
            language=language,
            commit_hash=commit_hash or "unknown",
        )
        user_prompt = load_prompt(
            "docstring_diff_user.txt",
            func_qualname=func_qualname,
            old_code=old_code or "(new function — no previous version)",
            new_code=new_code or "(deleted function)",
            commit_hash=commit_hash or "unknown",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self.stats.total_calls += 1

        try:
            result = self._call_llm(messages)
            parsed = self._parse_structured_response(result["content"])

            if not parsed["docstring"].strip():
                self.stats.failed_calls += 1
                self.stats.errors.append(f"{func_qualname}: empty response (diff)")
                return {
                    "success": False,
                    "docstring": "",
                    "change_analysis": "",
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
                "docstring": parsed["docstring"],
                "change_analysis": parsed["analysis"],
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
                "change_analysis": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": str(e),
            }

    def generate_docstring_with_impact(
        self,
        func_source: str,
        func_qualname: str,
        parent_qualname: str,
        parent_analysis: str,
        relationship: str,
        style: str = "google",
        language: str = "English",
        commit_hash: str = "",
    ) -> Dict:
        """
        Hop 1+ 專用：基於函式全文和上層分析結果生成 docstring。

        Args:
            func_source: 當前函式的完整源碼（僅函式，非整個檔案）
            func_qualname: 函式完整限定名
            parent_qualname: 上游被變更函式的限定名
            parent_analysis: 上游變更的分析摘要
            relationship: 與上游函式的關係描述（如 "calls parent_func"）
            style: docstring 風格
            language: 輸出語言
            commit_hash: commit SHA（用於在 docstring 的 Changelog 段落中引用）

        Returns:
            {
                "success": bool,
                "docstring": str,
                "impact_analysis": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": int,
                "error": Optional[str],
            }
        """
        style_desc = self._get_style_desc(style)

        system_prompt = load_prompt(
            "docstring_impact_system.txt",
            style_desc=style_desc,
            language=language,
            commit_hash=commit_hash or "unknown",
        )
        user_prompt = load_prompt(
            "docstring_impact_user.txt",
            func_qualname=func_qualname,
            func_source=func_source,
            parent_qualname=parent_qualname,
            parent_analysis=parent_analysis or "(no upstream analysis available)",
            relationship=relationship,
            commit_hash=commit_hash or "unknown",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self.stats.total_calls += 1

        try:
            result = self._call_llm(messages)
            parsed = self._parse_structured_response(result["content"])

            if not parsed["docstring"].strip():
                self.stats.failed_calls += 1
                self.stats.errors.append(f"{func_qualname}: empty response (impact)")
                return {
                    "success": False,
                    "docstring": "",
                    "impact_analysis": "",
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
                "docstring": parsed["docstring"],
                "impact_analysis": parsed["analysis"],
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
                "impact_analysis": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # README generation
    # ------------------------------------------------------------------

    def generate_readme(
        self,
        commit_hash: str,
        repo_name: str,
        seeds_summary: str,
        targets_summary: str,
        per_target_details: str,
        diff_text: str,
        language: str = "English",
    ) -> Dict:
        """
        生成 commit 級別的 README 文檔，描述本次 commit 的完整作用。

        Args:
            commit_hash: commit SHA
            repo_name: 倉庫名稱
            seeds_summary: 直接修改的 seeds 摘要文本
            targets_summary: 受影響的 targets 列表摘要
            per_target_details: 每個 target 的 docstring 更新和分析詳情
            diff_text: git diff 文本
            language: 輸出語言

        Returns:
            {
                "success": bool,
                "readme_content": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": int,
                "error": Optional[str],
            }
        """
        system_prompt = load_prompt(
            "readme_system.txt",
            language=language,
        )
        user_prompt = load_prompt(
            "readme_user.txt",
            commit_hash=commit_hash,
            repo_name=repo_name,
            seeds_summary=seeds_summary,
            targets_summary=targets_summary,
            per_target_details=per_target_details,
            diff_text=diff_text[:8000],  # 截斷過長的 diff
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self.stats.total_calls += 1

        # README 需要更長的輸出，臨時使用更高的 max_tokens
        saved_max_tokens = self.max_tokens
        self.max_tokens = max(self.max_tokens, 2048)

        try:
            result = self._call_llm(messages, temperature=0.3)
            content = result["content"].strip()

            # 移除 <think>...</think> 標籤
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            if not content:
                self.stats.failed_calls += 1
                return {
                    "success": False,
                    "readme_content": "",
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
                "readme_content": content,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
                "error": None,
            }

        except Exception as e:
            self.stats.failed_calls += 1
            self.stats.errors.append(f"readme_{commit_hash}: {str(e)}")
            return {
                "success": False,
                "readme_content": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": str(e),
            }
        finally:
            self.max_tokens = saved_max_tokens

    # ------------------------------------------------------------------
    # Module-level change analysis (fallback when targets == 0)
    # ------------------------------------------------------------------

    def _parse_affected_functions_response(self, content: str) -> List[Dict]:
        """
        解析 [AFFECTED_FUNCTIONS]...[/AFFECTED_FUNCTIONS] 結構化回應。

        Returns:
            list of {"qualname": str, "path": str, "reason": str}
            解析失敗時返回空列表。
        """
        text = content.strip()
        # 移除 <think>...</think> 標籤
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        m = re.search(
            r"\[AFFECTED_FUNCTIONS\]\s*(.*?)\s*\[/AFFECTED_FUNCTIONS\]",
            text,
            re.DOTALL,
        )
        if not m:
            return []

        json_text = m.group(1).strip()
        if not json_text:
            return []

        try:
            items = _json.loads(json_text)
            if not isinstance(items, list):
                return []
            # 校驗每個元素
            result: List[Dict] = []
            for it in items:
                if isinstance(it, dict) and "qualname" in it:
                    result.append({
                        "qualname": str(it["qualname"]),
                        "path": str(it.get("path", "")),
                        "reason": str(it.get("reason", "")),
                    })
            return result
        except (_json.JSONDecodeError, TypeError, ValueError):
            return []

    def analyze_module_level_changes(
        self,
        diff_text: str,
        file_sources: Dict[str, str],
        file_functions: Dict[str, List[str]],
        language: str = "English",
    ) -> Dict:
        """
        分析模塊級變更，識別受影響的函式。

        當 seed extractor 只產生 module-level seeds（kind="module"）且 target
        selector 未找到任何函式級 target 時，由此方法讓 LLM 分析 diff 並回傳
        受影響的函式列表。

        Args:
            diff_text: commit 的 unified diff 文本
            file_sources: {rel_path: 完整源碼} 已變更的 .py 檔案
            file_functions: {rel_path: [qualname, ...]} 每個檔案中的函式列表
            language: 輸出語言

        Returns:
            {
                "success": bool,
                "affected_functions": [{"qualname": str, "path": str, "reason": str}, ...],
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": int,
                "error": Optional[str],
            }
        """
        # 構建檔案源碼段
        file_section_parts: List[str] = []
        for rel_path, src in file_sources.items():
            file_section_parts.append(f"### {rel_path}\n```python\n{src}\n```")
        file_sections = "\n\n".join(file_section_parts) if file_section_parts else "(no Python files)"

        # 構建可用函式列表段
        func_section_parts: List[str] = []
        for rel_path, funcs in file_functions.items():
            if funcs:
                func_list = "\n".join(f"- {qn}" for qn in funcs)
                func_section_parts.append(f"### {rel_path}\n{func_list}")
            else:
                func_section_parts.append(f"### {rel_path}\n(no functions found)")
        function_sections = "\n\n".join(func_section_parts) if func_section_parts else "(no functions)"

        system_prompt = load_prompt(
            "module_analysis_system.txt",
            language=language,
        )
        user_prompt = load_prompt(
            "module_analysis_user.txt",
            diff_text=diff_text,
            file_sections=file_sections,
            function_sections=function_sections,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self.stats.total_calls += 1

        try:
            result = self._call_llm(messages)
            affected = self._parse_affected_functions_response(result["content"])

            # 過濾：只保留在 file_functions 中出現的 qualname
            valid_qualnames: set = set()
            for funcs in file_functions.values():
                valid_qualnames.update(funcs)

            affected = [a for a in affected if a["qualname"] in valid_qualnames]

            self.stats.success_calls += 1
            self.stats.total_prompt_tokens += result["prompt_tokens"]
            self.stats.total_completion_tokens += result["completion_tokens"]
            self.stats.total_latency_ms += result["latency_ms"]

            return {
                "success": True,
                "affected_functions": affected,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
                "error": None,
            }

        except Exception as e:
            self.stats.failed_calls += 1
            self.stats.errors.append(f"module_analysis: {str(e)}")
            return {
                "success": False,
                "affected_functions": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": str(e),
            }

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
