#!/usr/bin/env python3
"""Backward-compatible v5.5 inference entry point.

The implementation now delegates to the architecture-oriented app in
apps/run_inference_v2.py, so legacy commands keep working while the runtime
uses ports, factories, and the GameAgentFacade.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.run_inference_v2 import main as run_inference_main


def main():
    parser = argparse.ArgumentParser(description="黑神话悟空 AI 推理 v5.5")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--goal-id", type=int, default=0, help="Goal ID")
    parser.add_argument("--duration", type=int, default=60, help="推理时长（秒）")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument(
        "--capture",
        choices=["screen", "pyautogui"],
        default="screen",
        help="截图后端",
    )
    parser.add_argument(
        "--controller",
        choices=["vigem", "pydirect", "dry-run"],
        default="vigem",
        help="控制后端；控制问题解决后推荐 vigem",
    )
    parser.add_argument("--registry", default=None, help="可选 SQLite 运行记录路径")
    parser.add_argument("--run-name", default=None, help="可选运行名称")
    args = parser.parse_args()

    sys.argv = [
        sys.argv[0],
        "--model",
        args.model,
        "--goal-id",
        str(args.goal_id),
        "--duration",
        str(args.duration),
        "--conf-threshold",
        str(args.conf_threshold),
        "--device",
        args.device,
        "--capture",
        args.capture,
        "--controller",
        args.controller,
    ]
    if args.registry:
        sys.argv.extend(["--registry", args.registry])
    if args.run_name:
        sys.argv.extend(["--run-name", args.run_name])

    run_inference_main()


if __name__ == "__main__":
    main()
