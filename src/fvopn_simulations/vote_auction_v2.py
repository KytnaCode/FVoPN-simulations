import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from fvopn_simulations.types import Vector


@dataclass(frozen=True)
class VoteAuctionModelV2Config:
    budgets: np.ndarray[tuple[int], np.dtype[np.float64]]
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

    def __stealed_per_level(self, n: int):
        k = self.config.multiplication_factor

        return cast(float, 1 - (1 / k ** (n - 1)))

    def __stealed_per_vote(self, n: int):
        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        return cast(float, self.__stealed_per_level(n) / (alpha * k ** (n - 1)))

    def __number_of_levels(self):
        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        return math.ceil(math.log((len(self.config.budgets) * (k - 1)) / alpha + 1, k))

    def __steal_bonus(self, n: int):
        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        x = 0
        for i in range(self.__number_of_levels() - n):
            x += self.__stealed_per_level(i + n + 1)

        return cast(float, x / (alpha * (k ** (n + 1) - 1) / (k - 1)))

    def __expected_value_per_vote(self, n: int):
        return 1 - self.__stealed_per_vote(n) + self.__steal_bonus(n)

    def __level_price(self, n: int) -> float:
        if n == 1:
            return min(1, self.config.budgets.max())

        return self.__level_price(n - 1) / (
            self.__expected_value_per_vote(n)
            / self.__expected_value_per_vote(n - 1)
            * 3
        )

    def __free_levels(self, n: int) -> int:
        l = self.__number_of_levels()

        if n != l:
            return 0

        alpha = self.config.root_slots
        k = self.config.multiplication_factor

        N = math.trunc(cast(float, alpha * (k**l - 1) / (k - 1)))

        v = N - len(self.config.budgets)

        return v

    def run(self):
        alpha = self.config.root_slots
        k = self.config.multiplication_factor
        l = self.__number_of_levels()

        budgets = np.column_stack(
            (
                np.arange(len(self.config.budgets)),
                self.config.budgets,
                self.config.budgets,
            )
        )

        budgets = budgets[budgets[:, 2].argsort()]

        votes = np.zeros((budgets[:, 2].size, l))
        weights = np.array([self.__expected_value_per_vote(n + 1) for n in range(l)])

        for i in range(l):
            for j in range(cast(int, alpha * k**i) - self.__free_levels(i + 1)):
                p = self.__level_price(i + 1)

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

                budgets[selected, 2] -= min(
                    cast(float, budgets[selected, 2][0]),
                    cast(float, budgets[:, 1].mean() * 2),
                )

                votes[selected, i] += 1

        votes = votes @ weights

        votes = votes / votes.sum()

        return pd.DataFrame(
            np.column_stack((budgets, votes)),
            columns=["index", "start_budget", "budget", "votes"],
        )
