from typing import cast

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


@given(
    st.integers(100, 1000),
    st.integers(10, 30),
    st.integers(20, 20),
    st.integers(2, 30),
    st.one_of(st.floats(0.8, 2), st.floats(2, 100)),
)
def test_splitting_budget_between_identities_must_not_increase_voting_power(
    size: int,
    root_slots: int,
    multiplication_factor: int,
    identities: int,
    whale_budget: float,
):
    rng = np.random.default_rng(42)

    budgets = rng.beta(a=0.5, b=8, size=size)

    budgets[0] = whale_budget

    budgets[1 : identities + 1] = whale_budget / identities

    conf = VoteAuctionModelV2Config(
        budgets=budgets,
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    model = VoteAuctionModelV2(conf, rng)

    result = model.run()

    a = cast(float, result.loc[0, "votes"])
    b = result.loc[1 : identities + 1, "votes"].sum()

    assert a >= b or np.isclose(a, b)
