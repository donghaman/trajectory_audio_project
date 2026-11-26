import math
import pytest
from panner import calculate_gains


def gains(az=0, el=0, dist=1, r0=1, k=1, el_strength=1, law='-3'):
    return calculate_gains(az, el, dist, r0, k, el_strength, law)[0]


@pytest.mark.parametrize('az,expected', [(-30,'FL'),(30,'FR'),(-135,'RL'),(135,'RR')])
def test_cardinal_speaker_only(az, expected):
    g = gains(az)
    assert g[expected] == pytest.approx(1.0)
    assert sum(v for k,v in g.items() if k != expected) == pytest.approx(0.0)


def test_front_minus3db_pair():
    g = gains(0)
    assert g['FL'] == pytest.approx(1/math.sqrt(2), rel=1e-5)
    assert g['FR'] == pytest.approx(1/math.sqrt(2), rel=1e-5)


def test_gain_energy_conservation():
    g = gains(37, 23, 4)
    assert sum(v*v for v in g.values()) == pytest.approx(1.0, rel=1e-6)


def test_wrap_boundary():
    g1 = gains(179)
    g2 = gains(-179)
    assert sum((g1[k]-g2[k])**2 for k in g1) == pytest.approx(0.0, abs=0.03)


def test_overhead():
    g = gains(el=90)
    assert all(v == pytest.approx(0.5) for v in g.values())
