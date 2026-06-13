"""Policy factory."""


def build_policy(name: str, model_path, goal_id: int, device: str):
    if name == "v55":
        from policies.v55_policy import V55GoalConditionedPolicy

        return V55GoalConditionedPolicy(
            model_path=model_path,
            goal_id=goal_id,
            device=device,
        )
    if name == "v56-history":
        from policies.v56_history_policy import V56HistoryPolicy

        return V56HistoryPolicy(
            model_path=model_path,
            goal_id=goal_id,
            device=device,
        )
    if name == "v58-temporal":
        from policies.v58_temporal_policy import V58TemporalPolicy

        return V58TemporalPolicy(
            model_path=model_path,
            goal_id=goal_id,
            device=device,
        )
    raise ValueError(f"Unknown policy: {name}")
