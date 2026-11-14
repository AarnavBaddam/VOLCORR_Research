import numpy as np
import pandas as pd
import pytest

from volcorr.common import gk_variance, qlike
from volcorr.stress import count_episodes, primary_flags


def test_gk_hand_computed():
    # O=100 H=102 L=99 C=101: 0.5*ln(102/99)^2 - (2ln2-1)*ln(1.01)^2
    #   = 0.5*0.00089119940 - 0.38629436*0.0000990090840 = 0.00040735305
    v, floored = gk_variance(np.array([100.0]), np.array([102.0]), np.array([99.0]), np.array([101.0]))
    assert v[0] == pytest.approx(0.00040735305, abs=2e-9)
    assert not floored[0]


def test_gk_floor():
    # a flat bar gives 0 and is floored; a bar whose range equals its open-to-close move stays positive
    v, floored = gk_variance(np.array([100.0, 100.0]), np.array([100.0, 101.0]),
                             np.array([100.0, 100.0]), np.array([100.0, 101.0]))
    assert list(v) == [1e-8, 0.5 * np.log(1.01) ** 2 - (2 * np.log(2) - 1) * np.log(1.01) ** 2]
    assert list(floored) == [True, False]
    v, floored = gk_variance(np.array([100.0]), np.array([100.5]), np.array([99.9]), np.array([100.5]), 1e-8)
    assert v[0] >= 1e-8


def test_qlike():
    assert qlike(3.0, 3.0) == 0
    assert qlike(2.0, 1.0) == pytest.approx(2 - np.log(2) - 1)
    assert qlike(1.0, 2.0) == pytest.approx(0.5 - np.log(0.5) - 1)


def test_stress_rule_synthetic_vix():
    # 252 days at 10, then a jump to 20 that persists.
    # With j twenties in the 252-day lookback the 80th percentile stays below 20 until j = 52.
    vix = pd.Series([10.0] * 252 + [20.0] * 100)
    f = primary_flags(vix).values
    assert not f[:252].any()          # no full lookback, or not above threshold
    assert f[252:304].all()           # j = 0..51 twenties in the lookback
    assert not f[304:].any()          # threshold has caught up


def test_stress_rule_no_lookahead():
    rng = np.random.default_rng(0)
    vix = pd.Series(15 + rng.standard_normal(600).cumsum() * 0.3)
    base = primary_flags(vix)
    changed = vix.copy()
    changed.iloc[400:] += 50          # future shock
    assert (primary_flags(changed).iloc[:400] == base.iloc[:400]).all()
    changed = vix.copy()
    changed.iloc[399] += 50           # the origin's own close is excluded from its threshold
    thr = vix.rolling(252).quantile(0.8).shift(1)
    assert thr.iloc[399] == changed.rolling(252).quantile(0.8).shift(1).iloc[399]


def test_episode_counter():
    def ep(s):
        return count_episodes([c == "S" for c in s])
    assert ep("") == 0 and ep("ccc") == 0
    assert ep("SSS") == 1
    assert ep("S" + "c" * 5 + "S") == 1     # gap of 5 does not split
    assert ep("S" + "c" * 6 + "S") == 2     # gap of 6 does
    assert ep("ccSScSSccccccSccccccSc") == 3
