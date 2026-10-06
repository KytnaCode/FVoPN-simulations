import unittest
from typing import cast

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from . import Vector, VoteAuctionModel, VoteAuctionModelConfig


def random_budget(seed: int, size: int) -> Vector:
    rng = np.random.default_rng(seed)

    return rng.beta(0.5, 8, size=size) * 1000


class TestVoteAuctionModel(unittest.TestCase):
    @given(st.integers(1, 1000), st.integers(1, 10000), st.integers(2, 10000))
    def test_votes_must_sum_up_to_1(self, n: int, m: int, k: int):

        conf = VoteAuctionModelConfig(
            42,
            random_budget(42, n),
            m,
            k,
            5,
            0.2,
            lambda x, _: x**2,
            lambda x, _: cast(float, x ** (1 / 2)),
            lambda b, _: b.mean(),
        )

        model = VoteAuctionModel(conf)

        got = model.run()

        self.assertAlmostEqual(got.sum(), 1)
