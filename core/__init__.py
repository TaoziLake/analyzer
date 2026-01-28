"""
Core processing modules for seed extraction, call graph building,
and impact propagation.
"""

from .seed_extractor import extract_seeds, main as extract_seeds_main
from .call_graph import build_call_graph, CallGraph
from .impact_propagator import (
    propagate_impacts,
    propagate_impacts_typed,
    PropagationConfig,
    ImpactedNode,
    create_ablation_config,
    compute_impact_stats,
    impacted_to_json_list,
    _classify_qualname,
    resolve_seeds_path,
)
from .style_detector import detect_repo_style, detect_repo_style_cached

__all__ = [
    "extract_seeds",
    "extract_seeds_main",
    "build_call_graph",
    "CallGraph",
    "propagate_impacts",
    "propagate_impacts_typed",
    "PropagationConfig",
    "ImpactedNode",
    "create_ablation_config",
    "compute_impact_stats",
    "impacted_to_json_list",
    "_classify_qualname",
    "resolve_seeds_path",
    "detect_repo_style",
    "detect_repo_style_cached",
]
