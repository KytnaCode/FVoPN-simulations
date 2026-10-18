import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from .types import Vector


@dataclass(frozen=True)
class VoteAuctionModelConfig:
    Seed: int
    Budgets: Vector
    VotesPerSlot: int
    MultiplicationFactor: int
    RootSlots: int
    StartTopPercent: float
    VotePercentFn: Callable[[float, int], float]
    LevelPercentFn: Callable[[float, int], float]
    CostFn: Callable[[Vector, float], float]


class VoteAuctionModel:
    config: VoteAuctionModelConfig
    rng: np.random.Generator

    def __init__(self, config: VoteAuctionModelConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(self.config.Seed)

    def __calculate_number_of_levels(self):
        size = len(self.config.Budgets)

        k = self.config.MultiplicationFactor
        alpha = self.config.RootSlots

        levels = math.ceil(math.log(1 + (size * (k - 1)) / alpha, k))

        return levels

    def __get_n_level_slots(self, n: int, levels: int, empty_slots: int):
        k = self.config.MultiplicationFactor
        alpha = self.config.RootSlots

        slots = alpha * int(math.pow(k, n))

        if n == levels - 1:
            slots -= empty_slots

        return slots

    def __calculate_empty_slots(self, levels: int) -> int:
        size = len(self.config.Budgets)

        alpha = self.config.RootSlots
        k = self.config.MultiplicationFactor

        slots = alpha * int((math.pow(k, levels) - 1) // (k - 1))

        return slots - size

    def __weighted_votes(
        self,
        levels: int,
        empty_slots: int,
        votes: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ):
        a = self.config.RootSlots
        k = self.config.MultiplicationFactor
        m = self.config.VotesPerSlot

        votes /= min(a, len(votes))

        for i in range(levels - 1):
            f = a * cast(int, k**i) * m

            f += a * cast(int, k ** (i + 1))

            if i + 1 == levels - 1:
                f -= empty_slots

            votes[:, i:] *= 1 / f

            votes[:, i:] *= a * k**i

            votes[:, i] *= m

        return votes.sum(axis=1)

    def run(self):
        budgets = np.sort(self.config.Budgets, descending=True)

        levels = self.__calculate_number_of_levels()

        empty_slots = self.__calculate_empty_slots(levels)

        votes = np.zeros(budgets.shape + (levels,))

        for i in range(levels):
            for j in range(self.__get_n_level_slots(i, levels, empty_slots)):
                p = self.config.StartTopPercent

                p = self.config.LevelPercentFn(p, i)

                p = self.config.VotePercentFn(p, j)

                minTopIndex = math.floor(budgets.size * p)

                minTopValue = cast(float, budgets[minTopIndex])

                elegibleBudgets = budgets[budgets >= minTopValue]

                electionWeights = None

                if elegibleBudgets.sum() > 0:
                    electionWeights = (
                        budgets[budgets >= minTopValue]
                        / budgets[budgets >= minTopValue].sum()
                    )

                selected = self.rng.choice(
                    np.where(budgets >= minTopValue)[0],
                    p=electionWeights,
                    size=1,
                )

                budgets[selected] -= max(
                    self.config.CostFn(budgets, minTopValue), minTopValue
                )

                votes[selected, i] += 1

        return self.__weighted_votes(levels, empty_slots, votes)
