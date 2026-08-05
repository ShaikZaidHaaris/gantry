"""A cheap fitted policy, for asking whether data carries signal at all."""

from .probe import VERSION, FittedProbe, ProbeLearner, features

__all__ = ["VERSION", "FittedProbe", "ProbeLearner", "features"]
