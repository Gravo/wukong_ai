"""Command vocabulary for command-conditioned local corridor keeping."""

LCC_COMMANDS = {
    0: "KEEP_CENTER",
    1: "TURN_LEFT_SOON",
    2: "TURN_RIGHT_SOON",
    3: "AVOID_LEFT_WALL",
    4: "AVOID_RIGHT_WALL",
    5: "ENTER_LEFT_OPENING",
    6: "ENTER_RIGHT_OPENING",
    7: "RECOVER_FROM_STUCK",
}

LCC_COMMAND_IDS = {name: idx for idx, name in LCC_COMMANDS.items()}


def normalize_command_name(value: str) -> str:
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def command_id(value: str | int) -> int:
    if isinstance(value, int):
        if value not in LCC_COMMANDS:
            raise ValueError(f"Unknown LCC command id: {value}")
        return value
    normalized = normalize_command_name(value)
    if normalized not in LCC_COMMAND_IDS:
        raise ValueError(f"Unknown LCC command: {value}")
    return LCC_COMMAND_IDS[normalized]
