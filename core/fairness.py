from enum import Enum

class FairnessMeasure(Enum):
    DP = "dp"  # Demographic Parity
    CR = "cr"  # Custom-ratio # TODO: Implement this


class FairConfig:
    def __init__(self, k: int, fairness: FairnessMeasure = FairnessMeasure.DP):
        """
        Parameters
        ----------
        k : int
            Number of colors.
        fairness : FairnessMeasure
            Fairness measure.
        """
        self.fairness = fairness
        self.k = k