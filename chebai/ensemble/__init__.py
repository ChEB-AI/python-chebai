from ._consolidator import WeightedMajorityVoting
from ._controller import NoActivationCondition


class FullEnsembleWMV(NoActivationCondition, WeightedMajorityVoting):
    """Full Ensemble (no activation condition) with Weighted Majority Voting"""

    pass


__all__ = ["FullEnsembleWMV"]
