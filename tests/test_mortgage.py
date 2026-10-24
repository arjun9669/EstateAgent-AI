import pytest

from estate_agent.tools import ToolError
from estate_agent.tools.mortgage import max_loan_to_value, monthly_payment, mortgage_calculator


def test_monthly_payment_matches_standard_amortisation():
    # 100,000 at 6% over 25 years is 644.30 a month.
    assert monthly_payment(100_000, 6, 25) == pytest.approx(644.30, abs=0.01)
    assert monthly_payment(120_000, 0, 10) == pytest.approx(1000)


@pytest.mark.parametrize("price, buyer, first_home, off_plan, expected", [
    (2_000_000, "expat", True, False, 0.80),
    (6_000_000, "expat", True, False, 0.70),
    (2_000_000, "national", True, False, 0.85),
    (6_000_000, "national", True, False, 0.75),
    (2_000_000, "expat", False, False, 0.60),
    (2_000_000, "national", False, False, 0.65),
    (2_000_000, "national", True, True, 0.50),
])
def test_loan_to_value_caps(price, buyer, first_home, off_plan, expected):
    assert max_loan_to_value(price, buyer, first_home, off_plan) == expected


def test_defaults_to_minimum_down_payment():
    result = mortgage_calculator(property_price=2_000_000, annual_rate_pct=4.5)
    assert result["down_payment"] == 400_000
    assert result["loan_amount"] == 1_600_000
    assert result["meets_ltv_cap"] is True
    assert result["upfront_costs"] == {
        "dld_transfer_fee": 80_000,
        "mortgage_registration_fee": 4_000,
        "agency_commission_incl_vat": 42_000,
    }
    assert result["cash_needed_at_purchase"] == 400_000 + 80_000 + 4_000 + 42_000


def test_flags_down_payment_below_cap():
    result = mortgage_calculator(property_price=2_000_000, annual_rate_pct=4.5, down_payment=200_000)
    assert result["meets_ltv_cap"] is False
    assert result["minimum_down_payment"] == 400_000


def test_debt_burden_check():
    result = mortgage_calculator(property_price=125_000, annual_rate_pct=6, down_payment=25_000,
                                 monthly_income=1_000)
    assert result["monthly_payment"] == 644
    assert result["debt_burden_pct"] == 64.4
    assert result["within_debt_burden_cap"] is False


def test_cash_purchase_has_no_loan():
    result = mortgage_calculator(property_price=1_000_000, annual_rate_pct=5, down_payment=1_000_000)
    assert result["loan_amount"] == 0 and result["monthly_payment"] == 0
    assert result["upfront_costs"]["mortgage_registration_fee"] == 0


@pytest.mark.parametrize("kwargs, message", [
    ({"property_price": 0, "annual_rate_pct": 4}, "positive"),
    ({"property_price": 1e6, "annual_rate_pct": 25}, "between 0 and 20"),
    ({"property_price": 1e6, "annual_rate_pct": 4, "years": 30}, "UAE maximum"),
    ({"property_price": 1e6, "annual_rate_pct": 4, "buyer": "tourist"}, "buyer"),
    ({"property_price": 1e6, "annual_rate_pct": 4, "down_payment": 2e6}, "down_payment"),
    ({"property_price": 1e6, "annual_rate_pct": 4, "monthly_income": 0}, "monthly_income"),
])
def test_rejects_bad_input(kwargs, message):
    with pytest.raises(ToolError, match=message):
        mortgage_calculator(**kwargs)
