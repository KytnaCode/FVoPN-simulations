from typing import cast

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from fvopn_simulations import VoteAuctionModelV2, VoteAuctionModelV2Config


@settings(deadline=None)
@given(st.integers(100, 1000), st.integers(1, 30), st.integers(2, 30))
def test_bigger_budget_should_yield_to_more_voting_power(
    size: int, root_slots: int, multiplication_factor: int
):
    expected_min_corr = 0.40

    rng = np.random.default_rng(42)

    conf = VoteAuctionModelV2Config(
        budgets=rng.beta(a=0.5, b=8, size=size),
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    N = 10
    corr = 0

    model = VoteAuctionModelV2(conf, rng)

    for _ in range(N):
        result = model.run()

        corr += result["start_budget"].corr(result["votes"])

    corr /= N

    assert corr >= expected_min_corr


@given(
    st.integers(100, 500),
    st.integers(10, 30),
    st.integers(2, 2),
    st.integers(2, 2),
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

    def V(whale_budget: float, identities: int):
        b = np.copy(budgets)
        b[0:identities] = whale_budget / identities

        conf = VoteAuctionModelV2Config(
            budgets=b,
            root_slots=root_slots,
            multiplication_factor=multiplication_factor,
        )

        model = VoteAuctionModelV2(conf, rng)

        result = model.run()

        dBudget = 1 + abs(whale_budget - result.loc[0:identities, "budget"].sum())

        return result.loc[0:identities, "votes"].sum() / dBudget

    a = V(whale_budget, 1)

    b = V(whale_budget, identities)

    assert a >= b or np.isclose(a, b, atol=cast(float, 1 * 10 ** (-identities + 1)))
