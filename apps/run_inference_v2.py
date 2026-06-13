#!/usr/bin/env python3
"""Run inference through the DIP/facade architecture."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime_config import AgentRuntimeConfig
from services.agent_factory import build_agent
from storage.experiment_registry import ExperimentRegistry


def main():
    parser = argparse.ArgumentParser(description="Wukong AI inference")
    parser.add_argument("--model", required=True, help="Checkpoint path")
    parser.add_argument("--goal-id", type=int, default=0)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture", choices=["screen", "pyautogui"], default="screen")
    parser.add_argument("--controller", choices=["vigem", "pydirect", "dry-run"], default="vigem")
    parser.add_argument("--policy", choices=["v55", "v56-history", "v58-temporal"], default="v55")
    parser.add_argument("--step-delay", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--registry", default=None, help="Optional SQLite registry path")
    parser.add_argument("--telemetry", default=None, help="Optional CSV path for per-step predictions")
    parser.add_argument("--gate", choices=["none", "rule"], default="none")
    parser.add_argument("--recovery-threshold", type=float, default=0.35)
    parser.add_argument("--gate-forward-threshold", type=float, default=0.35)
    parser.add_argument("--gate-turn-dx", type=int, default=100)
    parser.add_argument("--gate-turn-hold-frames", type=int, default=4)
    parser.add_argument("--lcc-model", default=None, help="Optional LCC checkpoint for local correction fusion")
    parser.add_argument("--lcc-policy", choices=["v56-history", "v58-temporal"], default="v56-history")
    parser.add_argument("--lcc-command-id", type=int, default=0)
    parser.add_argument("--lcc-threshold", type=float, default=0.55)
    parser.add_argument("--lcc-override-frames", type=int, default=4)
    parser.add_argument(
        "--no-always-forward",
        action="store_true",
        help="Disable default forward movement for yaw-style policies.",
    )
    args = parser.parse_args()

    config = AgentRuntimeConfig(
        model_path=Path(args.model),
        goal_id=args.goal_id,
        duration=args.duration,
        conf_threshold=args.conf_threshold,
        device=args.device,
        capture=args.capture,
        controller=args.controller,
        policy=args.policy,
        step_delay=args.step_delay,
        log_every=args.log_every,
        run_name=args.run_name,
        registry_path=Path(args.registry) if args.registry else None,
        telemetry_path=Path(args.telemetry) if args.telemetry else None,
        always_forward=not args.no_always_forward,
        gate=args.gate,
        recovery_threshold=args.recovery_threshold,
        gate_forward_threshold=args.gate_forward_threshold,
        gate_turn_dx=args.gate_turn_dx,
        gate_turn_hold_frames=args.gate_turn_hold_frames,
        lcc_model_path=Path(args.lcc_model) if args.lcc_model else None,
        lcc_policy=args.lcc_policy,
        lcc_command_id=args.lcc_command_id,
        lcc_threshold=args.lcc_threshold,
        lcc_override_frames=args.lcc_override_frames,
    )

    registry = ExperimentRegistry(config.registry_path) if config.registry_path else None
    run_id = registry.create_run(config) if registry else None

    agent = build_agent(config)
    steps = agent.run_inference(
        duration=config.duration,
        log_every=config.log_every,
        telemetry_path=config.telemetry_path,
    )

    if registry and run_id is not None:
        registry.finish_run(run_id, steps)
        registry.close()

    executed = sum(1 for step in steps if step.executed)
    print(f"[agent] finished: executed={executed}, total_steps={len(steps)}", flush=True)


if __name__ == "__main__":
    main()
