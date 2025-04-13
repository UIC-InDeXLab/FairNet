from typing import List
from core.fairness import *
from algorithms.hittingset import HittingSetStrategy
from core.points import Point

def find_fair_hitting_set(
    strategy: HittingSetStrategy, fairconfig: FairConfig, **kwargs
) -> List[Point]:
    # if strategy == HittingSetStrategy.GREEDY:
    #     return find_fair_hitting_set_greedy(**kwargs)
    # elif strategy == HittingSetStrategy.GEOMETRIC:
    #     return find_fair_hitting_set_geometric(**kwargs)
    # else:
    #     raise NotImplementedError("Strategy not implemented.")
    # TODO: Implement the function
    pass