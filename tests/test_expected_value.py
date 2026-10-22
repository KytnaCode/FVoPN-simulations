import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from fvopn_simulations import VoteAuctionModelV2, VoteAuctionModelV2Config


@given(st.integers(1, 1000), st.integers(1, 30), st.integers(2, 40))
def test_expected_value_must_decrease_with_level(
    size: int, root_slots: int, multiplication_factor: int
):
    # amortization
    a = 0.1

    rng = np.random.default_rng(42)

    conf = VoteAuctionModelV2Config(
        budgets=rng.beta(a=0.5, b=8, size=size),
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    model = VoteAuctionModelV2(conf, rng)

    for i in range(model.number_of_levels() - 1):
        assert model.expected_value(i + 1) + a >= model.expected_value(i + 2)


@given(st.integers(1, 1000), st.integers(1, 30), st.integers(2, 40))
def test_expected_value_must_be_more_expensive_with_level(
    size: int, root_slots: int, multiplication_factor: int
):
    rng = np.random.default_rng(42)

    conf = VoteAuctionModelV2Config(
        budgets=rng.beta(a=0.5, b=8, size=size),
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    model = VoteAuctionModelV2(conf, rng)

    for i in range(model.number_of_levels() - 1):
        assert model.expected_value(i + 1) / model.level_price(
            i + 1
        ) >= model.expected_value(i + 2) / model.level_price(i + 2)
