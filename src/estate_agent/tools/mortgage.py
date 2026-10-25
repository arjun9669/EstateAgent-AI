"""mortgage_calculator: repayments, UAE loan-to-value caps and upfront costs.

calculate_mortgage() does the maths and returns figures with code-style keys
for programs and tests. mortgage_calculator(), the tool the agent sees, returns
the same figures with plain-English names and its checks as sentences: with raw
keys the model sometimes repeated names like "within_debt_burden_cap" in its
answers (docs/EVALUATION.md, v1 G4).

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


def calculate_mortgage(
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
    the whole term. Raises ToolError for invalid input.
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


def _buyer_description(buyer: str, first_home: bool, off_plan: bool, price: float) -> str:
    who = "an expatriate" if buyer == "expat" else "a UAE national"
    if off_plan:
        return "an off-plan property"
    if not first_home:
        return f"a second or later home for {who}"
    bracket = "up to" if price <= PRICE_BRACKET_AED else "above"
    return f"a first home {bracket} AED {PRICE_BRACKET_AED:,} for {who}"


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
    """The agent's tool: calculate_mortgage() in plain English, with checks as sentences."""
    m = calculate_mortgage(property_price, annual_rate_pct, down_payment, years, buyer,
                           first_home, off_plan, monthly_income)
    a = m["assumptions"]
    kind = _buyer_description(a["buyer"], a["first_home"], a["off_plan"], property_price)
    if m["meets_ltv_cap"]:
        ltv_check = (f"The down payment of AED {m['down_payment']:,} meets the minimum of "
                     f"AED {m['minimum_down_payment']:,}: banks can lend up to "
                     f"{m['max_loan_to_value_pct']:g}% of the price for {kind}.")
    else:
        ltv_check = (f"The down payment of AED {m['down_payment']:,} is below the minimum of "
                     f"AED {m['minimum_down_payment']:,}: banks can lend at most "
                     f"{m['max_loan_to_value_pct']:g}% of the price for {kind}, and this loan "
                     f"would be {m['loan_to_value_pct']:g}%.")
    checks = [ltv_check]
    result = {
        "source": m["source"],
        "monthly payment (AED)": m["monthly_payment"],
        "loan amount (AED)": m["loan_amount"],
        "down payment (AED)": m["down_payment"],
        "minimum down payment (AED)": m["minimum_down_payment"],
        "loan as % of price": m["loan_to_value_pct"],
        "maximum loan as % of price": m["max_loan_to_value_pct"],
        "total interest over the term (AED)": m["total_interest"],
        "upfront costs (AED)": {
            "DLD transfer fee (4% of price)": m["upfront_costs"]["dld_transfer_fee"],
            "mortgage registration fee (0.25% of loan)": m["upfront_costs"]["mortgage_registration_fee"],
            "agency commission (2% plus 5% VAT)": m["upfront_costs"]["agency_commission_incl_vat"],
        },
        "cash needed at purchase (AED)": m["cash_needed_at_purchase"],
    }
    if "debt_burden_pct" in m:
        result["payment as % of monthly income"] = m["debt_burden_pct"]
        position = "within" if m["within_debt_burden_cap"] else "above"
        checks.append(f"The monthly payment is {m['debt_burden_pct']:g}% of monthly income, {position} "
                      f"the {MAX_DEBT_BURDEN:.0%} debt burden limit. The limit covers all debt "
                      "repayments; this counts only this mortgage.")
    result["checks"] = checks
    result["assumptions"] = {
        "interest rate (% a year)": a["annual_rate_pct"],
        "term (years)": a["years"],
        "buyer": "expatriate" if a["buyer"] == "expat" else "UAE national",
        "first home": a["first_home"],
        "off-plan": a["off_plan"],
        "rate fixed for the whole term": True,
    }
    return result
