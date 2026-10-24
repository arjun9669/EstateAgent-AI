from estate_agent.areas import Area, area_names, resolve_area


class ToolError(ValueError):
    """Bad tool input. The message is shown to the agent so it can retry."""


def require_area(text: str) -> Area:
    area = resolve_area(text)
    if area is None:
        raise ToolError(f"Unknown area {text!r}. Known areas: {', '.join(area_names())}")
    return area


def require_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    value = value.lower()
    if value not in choices:
        raise ToolError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
    return value


def pct_change(new: float, old: float) -> float:
    return round((new - old) / old * 100, 1)
