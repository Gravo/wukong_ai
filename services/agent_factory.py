"""Composition root for game agent applications."""
from capture.factory import build_capture
from controllers.factory import build_controller
from policies.factory import build_policy
from services.game_agent_facade import GameAgentFacade
from services.rule_gate import RuleGate


def build_agent(config):
    """Wire the runtime object graph from a single config object."""
    policy = build_policy(
        name=config.policy,
        model_path=config.model_path,
        goal_id=config.goal_id,
        device=config.device,
    )
    if config.lcc_model_path:
        from policies.lcc_assist_policy import LccAssistPolicy

        lcc_policy = build_policy(
            name=config.lcc_policy,
            model_path=config.lcc_model_path,
            goal_id=config.lcc_command_id,
            device=config.device,
        )
        policy = LccAssistPolicy(
            route_policy=policy,
            lcc_policy=lcc_policy,
            lcc_threshold=config.lcc_threshold,
            override_frames=config.lcc_override_frames,
        )
    capture = build_capture(config.capture)
    controller = build_controller(
        config.controller,
        log_every=config.log_every,
        always_forward=config.always_forward,
    )
    gate = None
    if config.gate == "rule":
        gate = RuleGate(
            base_threshold=config.conf_threshold,
            recovery_threshold=config.recovery_threshold,
            forward_threshold=config.gate_forward_threshold,
            turn_dx=config.gate_turn_dx,
            turn_hold_frames=config.gate_turn_hold_frames,
        )
    return GameAgentFacade(
        capture=capture,
        policy=policy,
        controller=controller,
        conf_threshold=config.conf_threshold,
        step_delay=config.step_delay,
        gate=gate,
    )
