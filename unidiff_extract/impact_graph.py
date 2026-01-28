#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
impact_graph.py

Typed BFS + 權重衰減的影響傳播演算法

核心公式：
    score(v) = max_{p: seed→v} ∏_{e∈p} w(type(e)) · γ^|p|

實現：
    使用 priority queue（類似 Dijkstra，但乘法/取 max）

輸出：
    每個 impacted 節點包含：
    - score: 影響分數
    - hop: 跳數
    - best_path: 最佳路徑（可解釋性）
    - edge_types_on_path: 路徑上的邊類型
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from call_graph import CallGraph


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
    score: float
    hop: int
    reason: str
    source: Optional[str]  # 來源節點
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
            "score": round(self.score, 6),
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
# 核心演算法：Typed BFS + 權重衰減
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

def propagate_from_seeds(
    seeds: List[Dict],
    call_graph: CallGraph,
    gamma: float = DEFAULT_GAMMA,
    threshold: float = DEFAULT_THRESHOLD,
    max_hops: int = 4,
    edge_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, ImpactedNode]:
    """
    便捷函數：從種子節點傳播影響
    
    Args:
        seeds: 種子節點列表
        call_graph: 調用圖
        gamma: hop 衰減係數
        threshold: 分數閾值
        max_hops: 最大跳數
        edge_weights: 自定義邊權重（用於消融實驗）
    
    Returns:
        {qualname: ImpactedNode}
    """
    config = PropagationConfig(
        gamma=gamma,
        threshold=threshold,
        max_hops=max_hops,
    )
    if edge_weights:
        config.edge_weights.update(edge_weights)
    
    return propagate_impacts_typed(seeds, call_graph, config)


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
        # 所有邊權重相同
        return PropagationConfig(
            edge_weights={k: 0.7 for k in EDGE_WEIGHTS},
            gamma=DEFAULT_GAMMA,
        )
    
    elif ablation_type == "no_decay":
        # 無 hop 衰減
        return PropagationConfig(
            gamma=1.0,
            max_hops=3,  # 減少 max_hops 避免爆炸
        )
    
    elif ablation_type == "high_decay":
        # 高衰減
        return PropagationConfig(
            gamma=0.5,
        )
    
    elif ablation_type == "calls_only":
        # 只考慮 CALLS 邊
        return PropagationConfig(
            edge_weights={
                "CALLS": 0.7,
                "CALLED_BY": 0.6,
            },
        )
    
    else:  # "full" or default
        return PropagationConfig()


# ============================================================================
# 統計與輸出
# ============================================================================

def compute_impact_stats(impacted: Dict[str, ImpactedNode]) -> Dict:
    """
    計算影響傳播統計
    """
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
    """
    將 impacted 轉換為 JSON 可序列化的列表（按分數降序）
    """
    nodes = sorted(impacted.values(), key=lambda x: (-x.score, x.hop, x.qualname))
    return [node.to_dict() for node in nodes]


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import os
    from datetime import datetime
    
    from call_graph import build_call_graph
    
    ap = argparse.ArgumentParser(description="Typed BFS 影響傳播")
    ap.add_argument("--repo", required=True, help="Git 倉庫路徑")
    ap.add_argument("--seeds", required=True, help="Seeds JSON 檔案路徑")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help=f"Hop 衰減係數（默認 {DEFAULT_GAMMA}）")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"分數閾值（默認 {DEFAULT_THRESHOLD}）")
    ap.add_argument("--max-hops", type=int, default=4, help="最大跳數（默認 4）")
    ap.add_argument("--ablation", choices=["full", "uniform", "no_decay", "high_decay", "calls_only"], 
                    default="full", help="消融實驗類型")
    ap.add_argument("--out", default=None, help="輸出 JSON 路徑")
    args = ap.parse_args()
    
    # 讀取 seeds
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)
    seeds = seeds_data.get("seeds", [])
    
    # 構建調用圖
    print(f"構建調用圖: {args.repo}")
    cg = build_call_graph(args.repo)
    print(f"調用圖包含 {len(cg.all_functions)} 個函數")
    
    # 創建配置
    if args.ablation != "full":
        config = create_ablation_config(args.ablation)
        print(f"使用消融配置: {args.ablation}")
    else:
        config = PropagationConfig(
            gamma=args.gamma,
            threshold=args.threshold,
            max_hops=args.max_hops,
        )
    
    # 傳播
    print(f"從 {len(seeds)} 個種子開始傳播...")
    impacted = propagate_impacts_typed(seeds, cg, config)
    
    # 統計
    stats = compute_impact_stats(impacted)
    print(f"受影響節點: {stats['total_nodes']}")
    print(f"按跳數分佈: {stats['by_hop']}")
    
    # 輸出
    output = {
        "commit": seeds_data.get("commit", ""),
        "repo": args.repo,
        "config": config.to_dict(),
        "ablation": args.ablation,
        "num_seeds": len(seeds),
        "num_impacted": len(impacted),
        "stats": stats,
        "impacted": impacted_to_json_list(impacted),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    
    if args.out:
        out_path = args.out
    else:
        seeds_dir = os.path.dirname(args.seeds)
        seeds_basename = os.path.basename(args.seeds)
        out_basename = seeds_basename.replace("impacted_seeds_", "impacted_typed_")
        out_path = os.path.join(seeds_dir, out_basename)
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"輸出: {out_path}")
