#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0209版本
Main entry point for the analyzer package.

Provides a unified CLI interface for all analyzer operations.
"""

import argparse
import sys
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="Analyzer: Incremental documentation maintenance tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (LLM mode)
  python -m analyzer.main run --repo ./black --commit abc123 --mode llm

  # Full workflow (template mode)
  python -m analyzer.main run --repo ./black --commit abc123 --mode template

  # Extract seeds only
  python -m analyzer.main extract-seeds --repo ./black --commit abc123

  # Build call graph and propagate impacts
  python -m analyzer.main propagate --repo ./black --seeds seeds.json

  # Batch processing
  python -m analyzer.main batch --repo ./black --n 200
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run full workflow")
    run_parser.add_argument("--repo", required=True, help="Git repository path")
    run_parser.add_argument("--commit", required=True, help="Commit SHA")
    run_parser.add_argument("--mode", choices=["template", "llm"], default="template", help="Agent mode")
    run_parser.add_argument("--max-hops", type=int, default=2, help="Max propagation hops")
    run_parser.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="treesitter")
    run_parser.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    run_parser.add_argument("--limit-targets", type=int, default=50)
    run_parser.add_argument("--style", choices=["auto", "google", "numpy", "sphinx"], default="auto", help="Docstring style (LLM mode only)")
    run_parser.add_argument("--no-llm", action="store_true", help="Disable LLM (LLM mode only)")
    run_parser.add_argument("--run-parent", default=None, help="Output parent directory")
    
    # extract-seeds command
    extract_parser = subparsers.add_parser("extract-seeds", help="Extract seeds from git diff")
    extract_parser.add_argument("--repo", required=True, help="Git repository path")
    extract_parser.add_argument("--commit", required=True, help="Commit SHA")
    extract_parser.add_argument("--out", default=None, help="Output JSON path")
    
    # propagate command
    propagate_parser = subparsers.add_parser("propagate", help="Build call graph and propagate impacts")
    propagate_parser.add_argument("--repo", required=True, help="Git repository path")
    propagate_parser.add_argument("--seeds", required=True, help="Seeds JSON path")
    propagate_parser.add_argument("--max-hops", type=int, default=2)
    propagate_parser.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="treesitter")
    propagate_parser.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    propagate_parser.add_argument("--mode", choices=["simple", "typed"], default="simple", help="Propagation mode")
    propagate_parser.add_argument("--out", default=None, help="Output JSON path")
    
    # batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple commits")
    batch_parser.add_argument("--repo", required=True, help="Git repository path")
    batch_parser.add_argument("--commits-file", default=None, help="File with commit SHAs (one per line)")
    batch_parser.add_argument("--n", type=int, default=200, help="Number of commits from git log")
    batch_parser.add_argument("--max-hops", type=int, default=2)
    batch_parser.add_argument("--parser", choices=["auto", "ast", "treesitter"], default="treesitter")
    batch_parser.add_argument("--seed-scope", choices=["all", "code_only"], default="code_only")
    batch_parser.add_argument("--limit-targets", type=int, default=50)
    batch_parser.add_argument("--out", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "run":
        if args.mode == "llm":
            from .agents.llm_agent import main as llm_main
            sys.argv = [
                "llm_agent",
                "--repo", args.repo,
                "--commit", args.commit,
                "--max-hops", str(args.max_hops),
                "--parser", args.parser,
                "--seed-scope", args.seed_scope,
                "--limit-targets", str(args.limit_targets),
                "--style", args.style,
            ]
            if args.no_llm:
                sys.argv.append("--no-llm")
            if args.run_parent:
                sys.argv.extend(["--run-parent", args.run_parent])
            llm_main()
        else:
            from .agents.template_agent import main as template_main
            sys.argv = [
                "template_agent",
                "--repo", args.repo,
                "--commit", args.commit,
                "--max-hops", str(args.max_hops),
                "--parser", args.parser,
                "--seed-scope", args.seed_scope,
                "--limit-targets", str(args.limit_targets),
            ]
            if args.run_parent:
                sys.argv.extend(["--run-parent", args.run_parent])
            template_main()
    
    elif args.command == "extract-seeds":
        from .core.seed_extractor import main as extract_main
        sys.argv = ["extract_seeds", "--repo", args.repo, "--commit", args.commit]
        if args.out:
            sys.argv.extend(["--out", args.out])
        extract_main()
    
    elif args.command == "propagate":
        if args.mode == "typed":
            from .core.impact_propagator import main as typed_main
            sys.argv = [
                "impact_propagator",
                "--repo", args.repo,
                "--seeds", args.seeds,
                "--max-hops", str(args.max_hops),
                "--parser", args.parser,
            ]
            if args.out:
                sys.argv.extend(["--out", args.out])
            typed_main()
        else:
            # Use process_impacted_set logic
            import json
            import os
            from .core import build_call_graph, propagate_impacts
            from .core.impact_propagator import _classify_qualname, resolve_seeds_path
            from datetime import datetime
            
            seeds_path = resolve_seeds_path(args.seeds)
            with open(seeds_path, "r", encoding="utf-8") as f:
                seeds_data = json.load(f)
            
            seeds = seeds_data["seeds"]
            if args.seed_scope == "code_only":
                seeds = [s for s in seeds if s.get("seed_type") == "code_change"]
            
            print(f"Building call graph for {args.repo}...")
            call_graph = build_call_graph(args.repo, parser_mode=args.parser)
            print(f"Found {len(call_graph.all_functions)} functions in call graph")
            
            print(f"Propagating impacts from {len(seeds)} seeds...")
            impacted = propagate_impacts(seeds, call_graph, max_hops=args.max_hops, seed_scope=args.seed_scope)
            
            # Classify nodes
            for qn, item in impacted.items():
                is_internal, is_test, is_external = _classify_qualname(args.repo, qn)
                item["is_internal"] = is_internal
                item["is_test"] = is_test
                item["is_external"] = is_external
            
            output = {
                "commit": seeds_data.get("commit", ""),
                "parent": seeds_data.get("parent", ""),
                "repo": args.repo,
                "num_seeds": len(seeds_data["seeds"]),
                "num_seeds_used": len(seeds),
                "seed_scope": args.seed_scope,
                "num_impacted": len(impacted),
                "max_hops": args.max_hops,
                "call_graph_stats": {
                    "total_functions": len(call_graph.all_functions),
                    "total_call_relations": sum(len(v) for v in call_graph.functions.values()),
                },
                "impacted": list(impacted.values()),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            
            if args.out:
                out_path = os.path.abspath(args.out)
            else:
                seeds_dir = os.path.dirname(seeds_path)
                seeds_basename = os.path.basename(seeds_path)
                out_basename = seeds_basename.replace("impacted_seeds_", "impacted_set_")
                out_path = os.path.join(seeds_dir, out_basename)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"Wrote {out_path} with {len(impacted)} impacted functions")
    
    elif args.command == "batch":
        from .experiments.batch_runner import main as batch_main
        sys.argv = [
            "batch_runner",
            "--repo", args.repo,
            "--n", str(args.n),
            "--max-hops", str(args.max_hops),
            "--parser", args.parser,
            "--seed-scope", args.seed_scope,
            "--limit-targets", str(args.limit_targets),
        ]
        if args.commits_file:
            sys.argv.extend(["--commits-file", args.commits_file])
        if args.out:
            sys.argv.extend(["--out", args.out])
        batch_main()


if __name__ == "__main__":
    main()
