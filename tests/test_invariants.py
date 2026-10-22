import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_almost_equal

from fvopn_simulations import VoteAuctionModelV2, VoteAuctionModelV2Config


@given(st.integers(1, 500), st.integers(1, 20), st.integers(2, 30))
def test_voter_power_must_sum_up_to_1(
    size: int, root_slots: int, multiplication_factor: int
):
    rng = np.random.default_rng(42)

    conf = VoteAuctionModelV2Config(
        budgets=rng.beta(a=0.5, b=8, size=size),
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    model = VoteAuctionModelV2(conf, rng)

    result = model.run()

    assert_almost_equal(1, result["votes"].sum())


@given(st.integers(1, 1000), st.integers(1, 30), st.integers(2, 30))
def test_free_slots_must_be_equal_or_greater_than_zero_and_less_than_level_size(
    size: int, root_slots: int, multiplication_factor: int
):
    rng = np.random.default_rng(42)

    conf = VoteAuctionModelV2Config(
        budgets=rng.beta(a=0.5, b=8, size=size),
        root_slots=root_slots,
        multiplication_factor=multiplication_factor,
    )

    model = VoteAuctionModelV2(conf, rng)

    l = model.number_of_levels()
    alpha = conf.root_slots
    k = conf.multiplication_factor

    for i in range(l - 1):
        assert model.free_slots(i) == 0

    assert model.free_slots(l) >= 0
    assert model.free_slots(l) < alpha * k ** (l - 1)
