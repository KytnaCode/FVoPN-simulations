import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from fvopn_simulations.types import Vector


@dataclass(frozen=True)
class VoteAuctionModelV2Config:
    budgets: Vector
    multiplication_factor: int
    root_slots: int


class VoteAuctionModelV2:
    rng: np.random.Generator
    config: VoteAuctionModelV2Config

    def __init__(
        self, conf: VoteAuctionModelV2Config, rng: np.random.Generator | None
    ) -> None:
        self.config = conf
        self.rng = rng if rng != None else np.random.default_rng()

    def stolen_from_level(self, n: int):
        k = self.config.multiplication_factor

        return cast(float, 1 - (1 / k ** (n - 1)))

    def stolen_from_vote(self, n: int):
        return self.stolen_from_level(n) / self.nodes_on_level(n)

    def nodes_on_level(self, n: int) -> int:
        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        return cast(int, alpha * k ** (n - 1))

    def nodes_up_to_level(self, n: int) -> int:
        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        return cast(int, alpha * (k ** (n + 1) - 1) / (k - 1))

    def number_of_levels(self):
        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        return math.ceil(math.log((len(self.config.budgets) * (k - 1)) / alpha + 1, k))

    def stolen(self, n: int):
        x = 0

        for i in range(self.number_of_levels() - n):
            x += self.stolen_from_level(i + n + 1)

        return x / self.nodes_up_to_level(n)

    def expected_value(self, n: int):
        return 1 - self.stolen_from_vote(n) + self.stolen(n)

    def level_price(self, n: int) -> float:
        if n == 1:
            return 0.1

        p = (1 - self.level_price(n - 1)) * 0.8  # TODO: Improve this

        return -p + 1

    def free_slots(self, n: int) -> int:
        l = self.number_of_levels()

        if n != l:
            return 0

        N = math.trunc(cast(float, self.nodes_up_to_level(l - 1)))

        v = N - len(self.config.budgets)

        return v

    def run(self):
        l = self.number_of_levels()

        budgets = np.column_stack(
            (
                np.arange(len(self.config.budgets)),
                self.config.budgets,
                self.config.budgets,
            )
        )

        budgets = budgets[budgets[:, 2].argsort()]

        votes = np.zeros((budgets[:, 2].size, l))
        weights = np.array([self.expected_value(n + 1) for n in range(l)])

        for i in range(l):
            for _ in range(self.nodes_on_level(i + 1) - self.free_slots(i + 1)):
                p = self.level_price(i + 1)

                minTopIndex = min(
                    math.floor(budgets[:, 2].size * p), budgets[:, 2].size - 1
                )

                minTopValue = cast(float, budgets[minTopIndex, 2])

                elegibleBudgets = budgets[:, 2][budgets[:, 2] >= minTopValue]

                electionWeights = None

                total = cast(float, np.sum(budgets[:, 2][budgets[:, 2] >= minTopValue]))

                if elegibleBudgets.sum() > 0:
                    electionWeights = cast(
                        Vector, budgets[:, 2][budgets[:, 2] >= minTopValue] / total
                    )

                selected = self.rng.choice(
                    np.where(budgets[:, 2] >= minTopValue)[0],
                    p=electionWeights,
                    size=1,
                )

                # Empiric cost function.
                cost = cast(float, (budgets[selected, 2][0] / 2) / budgets[:, 2].sum())

                budgets[selected, 2] -= min(
                    cast(float, budgets[selected, 2][0]),
                    cost,
                )

                votes[selected, i] += 1

        votes = votes @ weights

        votes = votes / votes.sum()

        return pd.DataFrame(
            np.column_stack((budgets[:, 1], budgets[:, 2], votes)),
            columns=["start_budget", "budget", "votes"],
            index=budgets[:, 0],
        )
