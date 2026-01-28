#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Impact propagation module supporting both simple BFS and Typed BFS with weight decay.
"""

from __future__ import annotations

import glob
import heapq
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .call_graph import CallGraph, build_call_graph


# ============================================================================
# 邊類型權重配置（可消融實驗）
# ============================================================================

EDGE_WEIGHTS: Dict[str, float] = {
    # 調用關係
    "CALLS": 0.7,           # 函數調用
    "CALLED_BY": 0.6,       # 被調用（反向）
    
    # 介面相關（高權重：直接影響文檔）
    "EXPOSES_CLI": 0.95,    # 暴露 CLI 參數
    "READS_CONFIG": 0.9,    # 讀取配置
    "USES_ENV": 0.9,        # 使用環境變數
    
    # 依賴關係
    "DATA_DEP": 0.8,        # 資料依賴
    "CTRL_DEP": 0.75,       # 控制依賴
    
    # 文檔關係
    "DOCS": 0.85,           # 文檔描述關係
    
    # OOP 關係
    "OVERRIDES": 0.7,       # 覆寫方法
    "IMPLEMENTS": 0.7,      # 實作介面
    
    # 定義關係
    "DEFINES": 0.5,         # 模組定義（較低，傳播較廣）
}

# Hop 衰減係數
DEFAULT_GAMMA: float = 0.85

# 分數閾值（低於此值不納入 impacted set）
DEFAULT_THRESHOLD: float = 0.01


# ============================================================================
# 資料結構
# ============================================================================

@dataclass
class ImpactedNode:
    """受影響節點"""
    qualname: str
    score: float = 1.0
    hop: int = 0
    reason: str = "direct_change"
    source: Optional[str] = None  # 來源節點
    best_path: List[str] = field(default_factory=list)
    edge_types_on_path: List[str] = field(default_factory=list)
    
    # 額外資訊
    callers: List[str] = field(default_factory=list)
    callees: List[str] = field(default_factory=list)
    is_internal: bool = False
    is_test: bool = False
    is_external: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "qualname": self.qualname,
            "score": round(self.score, 6) if isinstance(self.score, float) else self.score,
            "hop": self.hop,
            "reason": self.reason,
            "source": self.source,
            "best_path": self.best_path,
            "edge_types_on_path": self.edge_types_on_path,
            "callers": self.callers,
            "callees": self.callees,
            "is_internal": self.is_internal,
            "is_test": self.is_test,
            "is_external": self.is_external,
        }


@dataclass
class PropagationConfig:
    """傳播配置"""
    edge_weights: Dict[str, float] = field(default_factory=lambda: EDGE_WEIGHTS.copy())
    gamma: float = DEFAULT_GAMMA
    threshold: float = DEFAULT_THRESHOLD
    max_hops: int = 4
    
    def to_dict(self) -> Dict:
        return {
            "edge_weights": self.edge_weights,
            "gamma": self.gamma,
            "threshold": self.threshold,
            "max_hops": self.max_hops,
        }


# ============================================================================
# 簡單 BFS 傳播（等權重）
# ============================================================================

def propagate_impacts(
    seeds: List[Dict],
    call_graph: CallGraph,
    max_hops: int = 2,
    seed_scope: str = "all",
) -> Dict[str, Dict]:
    """
    簡單 BFS 傳播（等權重版本）
    
    Args:
        seeds: 種子節點列表
        call_graph: 調用圖
        max_hops: 最大跳數
        seed_scope: "all" 或 "code_only"
    
    Returns:
        {qualname: Dict} - 每個 impacted 節點的字典表示
    """
    # 過濾 seeds
    seeds_used = seeds
    if seed_scope == "code_only":
        seeds_used = [s for s in seeds if s.get("seed_type") == "code_change"]
    
    impacted: Dict[str, Dict] = {}
    visited: Set[str] = set()
    queue: List[Tuple[str, int]] = []

    def add_impacted(qualname: str, hop: int, reason: str, source_qualname: Optional[str]):
        if qualname not in impacted:
            impacted[qualname] = {
                "qualname": qualname,
                "hop": hop,
                "reason": reason,
                "source": source_qualname,
                "callers": list(call_graph.reverse_functions.get(qualname, set())),
                "callees": list(call_graph.functions.get(qualname, set())),
            }
        elif hop < impacted[qualname]["hop"]:
            impacted[qualname]["hop"] = hop
            impacted[qualname]["reason"] = reason
            impacted[qualname]["source"] = source_qualname

    for seed in seeds_used:
        qn = seed.get("qualname")
        if not qn:
            continue
        if qn in call_graph.all_functions:
            add_impacted(qn, 0, "direct_change", None)
            queue.append((qn, 0))
            visited.add(qn)

    while queue:
        cur, hop = queue.pop(0)
        if hop >= max_hops:
            continue
        nxt = hop + 1

        for caller in call_graph.reverse_functions.get(cur, set()):
            if caller not in visited:
                add_impacted(caller, nxt, "calls_changed_function", cur)
                queue.append((caller, nxt))
                visited.add(caller)

        for callee in call_graph.functions.get(cur, set()):
            if callee not in visited:
                add_impacted(callee, nxt, "called_by_changed_function", cur)
                queue.append((callee, nxt))
                visited.add(callee)

    return impacted


# ============================================================================
# Typed BFS + 權重衰減
# ============================================================================

def propagate_impacts_typed(
    seeds: List[Dict],
    call_graph: CallGraph,
    config: Optional[PropagationConfig] = None,
) -> Dict[str, ImpactedNode]:
    """
    帶權重衰減的影響傳播演算法
    
    使用 priority queue（類似 Dijkstra，但乘法/取 max）
    
    Args:
        seeds: 種子節點列表，每個元素需包含 "qualname"
        call_graph: 調用圖
        config: 傳播配置
        
    Returns:
        {qualname: ImpactedNode}
    """
    if config is None:
        config = PropagationConfig()
    
    # 結果集
    impacted: Dict[str, ImpactedNode] = {}
    
    # Priority queue: (-score, hop, qualname, path, edge_types)
    # 使用負分數是因為 heapq 是 min-heap
    pq: List[Tuple[float, int, str, List[str], List[str]]] = []
    
    # 記錄已處理的最佳分數
    best_scores: Dict[str, float] = {}
    
    # 初始化種子節點
    for seed in seeds:
        qn = seed.get("qualname", "")
        if not qn:
            continue
        
        # 種子節點只要在調用圖中有記錄就納入
        if qn in call_graph.all_functions:
            initial_score = 1.0
            heapq.heappush(pq, (-initial_score, 0, qn, [qn], []))
            best_scores[qn] = initial_score
    
    # Dijkstra-style 傳播
    while pq:
        neg_score, hop, current, path, edge_types = heapq.heappop(pq)
        current_score = -neg_score
        
        # 跳過已經有更好分數的節點
        if current in impacted and impacted[current].score >= current_score:
            continue
        
        # 記錄當前節點
        is_seed = (hop == 0)
        impacted[current] = ImpactedNode(
            qualname=current,
            score=current_score,
            hop=hop,
            reason="direct_change" if is_seed else "propagated",
            source=path[-2] if len(path) >= 2 else None,
            best_path=path.copy(),
            edge_types_on_path=edge_types.copy(),
            callers=list(call_graph.reverse_functions.get(current, set())),
            callees=list(call_graph.functions.get(current, set())),
        )
        
        # 檢查是否達到最大跳數
        if hop >= config.max_hops:
            continue
        
        # 擴展鄰居
        next_hop = hop + 1
        
        # 1. CALLED_BY 邊：當前節點被誰調用
        for caller in call_graph.reverse_functions.get(current, set()):
            _try_extend(
                pq, best_scores, config,
                caller, current_score, next_hop,
                path + [caller], edge_types + ["CALLED_BY"],
                "CALLED_BY"
            )
        
        # 2. CALLS 邊：當前節點調用誰
        for callee in call_graph.functions.get(current, set()):
            _try_extend(
                pq, best_scores, config,
                callee, current_score, next_hop,
                path + [callee], edge_types + ["CALLS"],
                "CALLS"
            )
    
    return impacted


def _try_extend(
    pq: List,
    best_scores: Dict[str, float],
    config: PropagationConfig,
    target: str,
    current_score: float,
    next_hop: int,
    new_path: List[str],
    new_edge_types: List[str],
    edge_type: str,
) -> None:
    """
    嘗試擴展到目標節點
    
    新分數 = 當前分數 × w(edge_type) × γ
    """
    weight = config.edge_weights.get(edge_type, 0.5)
    new_score = current_score * weight * config.gamma
    
    # 低於閾值，不擴展
    if new_score < config.threshold:
        return
    
    # 如果已有更好分數，不擴展
    if target in best_scores and best_scores[target] >= new_score:
        return
    
    # 更新最佳分數並入隊
    best_scores[target] = new_score
    heapq.heappush(pq, (-new_score, next_hop, target, new_path, new_edge_types))


# ============================================================================
# 便捷函數
# ============================================================================

def create_ablation_config(ablation_type: str) -> PropagationConfig:
    """
    創建消融實驗配置
    
    Args:
        ablation_type: 消融類型
            - "uniform": 所有邊權重相同（等權重 baseline）
            - "no_decay": 無 hop 衰減（γ=1.0）
            - "high_decay": 高衰減（γ=0.5）
            - "calls_only": 只考慮 CALLS 邊
            - "full": 完整配置（對照組）
    
    Returns:
        PropagationConfig
    """
    if ablation_type == "uniform":
        return PropagationConfig(
            edge_weights={k: 0.7 for k in EDGE_WEIGHTS},
            gamma=DEFAULT_GAMMA,
        )
    elif ablation_type == "no_decay":
        return PropagationConfig(
            gamma=1.0,
            max_hops=3,
        )
    elif ablation_type == "high_decay":
        return PropagationConfig(
            gamma=0.5,
        )
    elif ablation_type == "calls_only":
        return PropagationConfig(
            edge_weights={
                "CALLS": 0.7,
                "CALLED_BY": 0.6,
            },
        )
    else:  # "full" or default
        return PropagationConfig()


def compute_impact_stats(impacted: Dict[str, ImpactedNode]) -> Dict:
    """計算影響傳播統計"""
    if not impacted:
        return {
            "total_nodes": 0,
            "by_hop": {},
            "by_edge_type": {},
            "score_distribution": {},
        }
    
    # 按 hop 統計
    by_hop: Dict[int, int] = {}
    for node in impacted.values():
        by_hop[node.hop] = by_hop.get(node.hop, 0) + 1
    
    # 按邊類型統計
    by_edge_type: Dict[str, int] = {}
    for node in impacted.values():
        for et in node.edge_types_on_path:
            by_edge_type[et] = by_edge_type.get(et, 0) + 1
    
    # 分數分佈
    scores = [node.score for node in impacted.values()]
    score_distribution = {
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(sum(scores) / len(scores), 6),
        "above_0.5": sum(1 for s in scores if s >= 0.5),
        "above_0.3": sum(1 for s in scores if s >= 0.3),
        "above_0.1": sum(1 for s in scores if s >= 0.1),
    }
    
    return {
        "total_nodes": len(impacted),
        "by_hop": dict(sorted(by_hop.items())),
        "by_edge_type": dict(sorted(by_edge_type.items(), key=lambda x: -x[1])),
        "score_distribution": score_distribution,
    }


def impacted_to_json_list(impacted: Dict[str, ImpactedNode]) -> List[Dict]:
    """將 impacted 轉換為 JSON 可序列化的列表（按分數降序）"""
    nodes = sorted(impacted.values(), key=lambda x: (-x.score, x.hop, x.qualname))
    return [node.to_dict() for node in nodes]


# ============================================================================
# 分類輔助函數
# ============================================================================

def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    base = os.path.basename(p)
    return ("/tests/" in p) or p.startswith("tests/") or base.startswith("test_")


def _classify_qualname(repo_path: str, qualname: str, ext: str = ".py") -> Tuple[bool, bool, bool]:
    """
    Heuristic classification:
      - internal: qualname can be mapped back to a repo module file (*.py or package/__init__.py) and not test
      - test:     mapped module file is under tests/ or test_*.py
      - external: cannot map back to repo source file
    """
    if not qualname:
        return False, False, True

    parts = [p for p in qualname.split(".") if p]
    if not parts:
        return False, False, True

    for i in range(len(parts), 0, -1):
        module_qual = ".".join(parts[:i])
        rel_mod = module_qual.replace(".", os.sep)

        cand1 = os.path.join(repo_path, f"{rel_mod}{ext}")
        cand2 = os.path.join(repo_path, rel_mod, f"__init__{ext}")

        if os.path.isfile(cand1):
            is_test = _is_test_path(os.path.relpath(cand1, repo_path))
            is_internal = not is_test
            return is_internal, is_test, False
        if os.path.isfile(cand2):
            is_test = _is_test_path(os.path.relpath(cand2, repo_path))
            is_internal = not is_test
            return is_internal, is_test, False

    return False, False, True


def resolve_seeds_path(seeds_arg: str) -> str:
    """
    解析 seeds 路徑（支持 glob pattern 和目錄）
    """
    if os.path.isfile(seeds_arg):
        return seeds_arg

    if any(ch in seeds_arg for ch in ("*", "?", "[")):
        matches = glob.glob(seeds_arg)
        if not matches:
            raise FileNotFoundError(f"seeds pattern 未匹配到文件: {seeds_arg}")
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chosen = matches[0]
        if len(matches) > 1:
            print(f"提示：seeds pattern 匹配到 {len(matches)} 個文件，默認取最新的：{chosen}")
        return chosen

    if os.path.isdir(seeds_arg):
        pattern = os.path.join(seeds_arg, "impacted_seeds_*.json")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"目錄下未找到 seeds 文件（{pattern}）")
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chosen = matches[0]
        print(f"提示：seeds 指向目錄，默認取最新的：{chosen}")
        return chosen

    raise FileNotFoundError(f"seeds 參數不是有效文件/目錄: {seeds_arg}")
