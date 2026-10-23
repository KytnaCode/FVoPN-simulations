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
        return self.stolen_from_level(n)

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
            return 0.2

        p = (1 - self.level_price(n - 1)) * 0.5

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

        stop = False

        for i in range(l):
            if stop:
                break

            for _ in range(self.nodes_on_level(i + 1) - self.free_slots(i + 1)):
                p = self.level_price(i + 1)

                budgets = budgets[budgets[:, 2].argsort()]

                minTopValue = cast(float, budgets[:, 2].max() * (1 - p))

                elegibleBudgets = budgets[budgets[:, 2] >= minTopValue]

                if len(elegibleBudgets) == 0:
                    elegibleBudgets = budgets[0:1]

                    if elegibleBudgets[0, 2] == 0:
                        stop = True
                        break

                budgets = budgets[budgets[:, 0].argsort()]

                total = cast(float, np.sum(elegibleBudgets[:, 2]))

                electionWeights = cast(Vector, elegibleBudgets[:, 2] / total)

                selected = self.rng.choice(  # pyright: ignore [reportAny]
                    elegibleBudgets,
                    p=electionWeights,
                    size=1,
                )[0]

                selInd: int = math.trunc(cast(float, selected[0]))

                budgets = budgets[budgets[:, 0].argsort()]

                # Empiric cost function.
                cost = cast(
                    float,
                    minTopValue * (1 - budgets[selInd, 2] / budgets[:, 2].max()),
                )

                cost = min(cost, cast(float, budgets[selInd, 2]))

                budgets[selInd, 2] -= cost

                votes[selInd, i] += 1

        votes = votes @ weights

        votes = votes / votes.sum()

        return pd.DataFrame(
            np.column_stack((budgets[:, 1], budgets[:, 2], votes)),
            columns=["start_budget", "budget", "votes"],
            index=budgets[:, 0],
        )
