import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from fvopn_simulations import VoteAuctionModelV2, VoteAuctionModelV2Config


@given(st.integers(100, 1000), st.integers(1, 30), st.integers(2, 30))
def test_bigger_budget_should_yield_to_more_voting_power(
    size: int, root_slots: int, multiplication_factor: int
):
    expected_min_corr = 0.8

    rng = np.random.default_rng(42)

    conf = VoteAuctionModelV2Config(
        budgets=rng.beta(a=0.5, b=8, size=size),
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    model = VoteAuctionModelV2(conf, rng)

    result = model.run()

    corr = result["start_budget"].corr(result["votes"])

    assert corr >= expected_min_corr
