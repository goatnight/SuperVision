"""运行工程 Benchmark 并把完整 JSON 结果写入指定目录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spinevision_agent.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SpineVision-Agent engineering benchmark.")
    parser.add_argument("--output-dir", default="agent_artifacts/benchmark")
    parser.add_argument("--real-cases", type=int, default=5)
    parser.add_argument("--real-slices", type=int, default=3)
    args = parser.parse_args()
    results = run_benchmark(Path(args.output_dir), real_cases=args.real_cases, real_slices=args.real_slices)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
