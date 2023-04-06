import pytest
from nightshift import coverage
import numpy as np


def test_coverage():
    H = 4
    D = 20
    dt = 20 / 60 / 24

    times = np.hstack([np.arange(0, H / 24, dt) + i for i in np.arange(D)])

    P_max = 10
    periods = np.arange(0, P_max, 0.01)
    covs = coverage(times, periods)
