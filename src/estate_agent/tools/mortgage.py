"""mortgage_calculator: repayments, UAE loan-to-value caps and upfront costs.

Rule values are listed with their sources in docs/DATA_SOURCES.md. Re-check
them before relying on results; lenders may apply stricter limits.
"""

from estate_agent.tools.base import ToolError, require_choice

SOURCE = (
    "UAE Central Bank mortgage rules and Dubai Land Department fees, checked October 2026; "
    "lenders may apply stricter limits"
)
BUYERS = ("expat", "national")
PRICE_BRACKET_AED = 5_000_000
# (buyer, first home, price above bracket) -> maximum loan-to-value
MAX_LTV = {
    ("expat", True, False): 0.80,
    ("expat", True, True): 0.70,
    ("national", True, False): 0.85,
    ("national", True, True): 0.75,
}
MAX_LTV_LATER_HOME = {"expat": 0.60, "national": 0.65}
MAX_LTV_OFF_PLAN = 0.50
MAX_YEARS = 25
MAX_DEBT_BURDEN = 0.50
DLD_TRANSFER_FEE = 0.04
MORTGAGE_REGISTRATION_FEE = 0.0025
AGENCY_COMMISSION = 0.02
VAT = 0.05


def max_loan_to_value(price: float, buyer: str, first_home: bool, off_plan: bool) -> float:
    if off_plan:
        return MAX_LTV_OFF_PLAN
    if not first_home:
        return MAX_LTV_LATER_HOME[buyer]
    return MAX_LTV[(buyer, True, price > PRICE_BRACKET_AED)]


def monthly_payment(loan: float, annual_rate_pct: float, years: int) -> float:
    months = years * 12
    rate = annual_rate_pct / 100 / 12
    if rate == 0:
        return loan / months
    return loan * rate / (1 - (1 + rate) ** -months)


def mortgage_calculator(
    property_price: float,
    annual_rate_pct: float,
    down_payment: float | None = None,
    years: int = 25,
    buyer: str = "expat",
    first_home: bool = True,
    off_plan: bool = False,
    monthly_income: float | None = None,
) -> dict:
    """Monthly repayment, loan-to-value check, upfront cash and optional affordability.

    down_payment defaults to the minimum allowed. The rate is assumed fixed for
    the whole term.
    """
    if property_price <= 0:
        raise ToolError("property_price must be positive")
    if not 0 <= annual_rate_pct <= 20:
        raise ToolError("annual_rate_pct must be between 0 and 20")
    if not 1 <= years <= MAX_YEARS:
        raise ToolError(f"years must be between 1 and {MAX_YEARS} (the UAE maximum term)")
    buyer = require_choice("buyer", buyer, BUYERS)

    cap = max_loan_to_value(property_price, buyer, first_home, off_plan)
    min_down = property_price * (1 - cap)
    if down_payment is None:
        down_payment = min_down
    if not 0 <= down_payment <= property_price:
        raise ToolError("down_payment must be between 0 and property_price")

    loan = property_price - down_payment
    payment = monthly_payment(loan, annual_rate_pct, years) if loan else 0.0
    upfront = {
        "dld_transfer_fee": property_price * DLD_TRANSFER_FEE,
        "mortgage_registration_fee": loan * MORTGAGE_REGISTRATION_FEE,
        "agency_commission_incl_vat": property_price * AGENCY_COMMISSION * (1 + VAT),
    }
    result = {
        "source": SOURCE,
        "loan_amount": round(loan),
        "down_payment": round(down_payment),
        "loan_to_value_pct": round(loan / property_price * 100, 1),
        "max_loan_to_value_pct": round(cap * 100, 1),
        # Compare in whole dirhams so the default minimum is never flagged.
        "meets_ltv_cap": round(down_payment) >= round(min_down),
        "minimum_down_payment": round(min_down),
        "monthly_payment": round(payment),
        "total_interest": round(payment * years * 12 - loan),
        "upfront_costs": {k: round(v) for k, v in upfront.items()},
        "cash_needed_at_purchase": round(down_payment + sum(upfront.values())),
        "assumptions": {
            "annual_rate_pct": annual_rate_pct,
            "years": years,
            "buyer": buyer,
            "first_home": first_home,
            "off_plan": off_plan,
            "fixed_rate_for_whole_term": True,
        },
    }
    if monthly_income is not None:
        if monthly_income <= 0:
            raise ToolError("monthly_income must be positive")
        ratio = payment / monthly_income
        result["debt_burden_pct"] = round(ratio * 100, 1)
        result["within_debt_burden_cap"] = ratio <= MAX_DEBT_BURDEN
        result["debt_burden_note"] = (
            f"The cap is {MAX_DEBT_BURDEN:.0%} of income across all debt repayments; "
            "this counts only this mortgage"
        )
    return result
