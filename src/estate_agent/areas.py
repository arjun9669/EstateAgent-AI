"""Dubai communities known to the tools, with common short names.

Price levels drive the synthetic data generator only. They are rough
illustrative figures, not market data.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Area:
    name: str
    aliases: tuple[str, ...]
    sale_price_per_sqft: int  # AED, illustrative
    gross_yield: float  # annual rent / sale price, illustrative
    property_types: tuple[str, ...]


AREAS: tuple[Area, ...] = (
    Area("Palm Jumeirah", ("palm", "the palm"), 3500, 0.050, ("apartment", "villa")),
    Area("Downtown Dubai", ("downtown",), 3000, 0.055, ("apartment",)),
    Area("Jumeirah Beach Residence", ("jbr",), 2400, 0.060, ("apartment",)),
    Area("Dubai Marina", ("marina",), 2200, 0.062, ("apartment",)),
    Area("Dubai Hills Estate", ("dubai hills",), 2100, 0.055, ("apartment", "villa", "townhouse")),
    Area("Business Bay", ("bb",), 2000, 0.065, ("apartment",)),
    Area("Arabian Ranches", ("ranches",), 1600, 0.050, ("villa", "townhouse")),
    Area("Jumeirah Lake Towers", ("jlt",), 1500, 0.070, ("apartment",)),
    Area("Al Barsha", ("barsha",), 1300, 0.065, ("apartment", "villa")),
    Area("Jumeirah Village Circle", ("jvc",), 1200, 0.075, ("apartment", "townhouse")),
    Area("Mirdif", (), 1100, 0.060, ("apartment", "villa")),
    Area("Dubai Sports City", ("sports city",), 1000, 0.075, ("apartment",)),
    Area("Dubai Silicon Oasis", ("dso", "silicon oasis"), 950, 0.075, ("apartment",)),
    Area("Deira", (), 900, 0.070, ("apartment",)),
    Area("International City", (), 700, 0.085, ("apartment",)),
)

_LOOKUP = {a.name.lower(): a for a in AREAS} | {
    alias: a for a in AREAS for alias in a.aliases
}


def resolve_area(text: str) -> Area | None:
    """Match an area by full name or alias, ignoring case and extra spaces."""
    return _LOOKUP.get(" ".join(text.lower().split()))


def area_names() -> list[str]:
    return [a.name for a in AREAS]
