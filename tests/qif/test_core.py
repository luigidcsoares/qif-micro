from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
import pytest

from qif_micro import qif
from qif_micro.qif.datatypes import Channel, Joint, ProbabDist, Strategy, StochMatrix

_CHANNEL_0 = np.array([
    [1/4, 1/2, 1/4],
    [0,   1,   0],
    [0,   0,   1]
])

_CHANNEL_1 = np.array([
    [0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0]
])

_JOINT_0 = np.array([
    [0.0625, 0.125, 0.0625],
    [0.0,    0.5,   0.0],
    [0.0,    0.0,   0.25]
])

_JOINT_1 = np.array([
    [1/6, 1/6, 0.0],
    [1/6, 1/6, 0.0],
    [0.0, 0.0, 1/3]
])

_OUTER_0 = [0.0625, 0.625, 0.3125]
_POSTERIOR_0 = np.array([
    [1.0, 0.2, 0.2],
    [0.0, 0.8, 0.0],
    [0.0, 0.0, 0.8]
])

_PRIOR_0 = np.array([1/4, 1/2, 1/4])
_PRIOR_1 = np.array([2/5, 1/5, 2/5])
_PRIOR_2 = np.array([1/3, 1/3, 1/3])

_STRATEGY_PRIOR_0 = np.array([0.0, 1.0, 0.0])
_STRATEGY_PRIOR_1 = np.array([0.5, 0.0, 0.5])

_STRATEGY_JOINT_0 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

_STRATEGY_JOINT_1 = np.array([
    [0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0]
])

# Partitioned channel data (sequence of 2D matrices)
# Partitioned by output domain: each partition is a slice of outputs
# All rows must sum to 1.0 ACROSS all partitions (complete channels)
_CHANNEL_PART_0 = np.array([
    [0.6, 0.4],      # input 0: partition sum = 1.0
    [0.3, 0.2],      # input 1: partition sum = 0.5
    [0.0, 0.0]       # input 2: partition sum = 0.0
])

_CHANNEL_PART_1 = np.array([
    [0.0, 0.0],      # input 0: partition sum = 0.0
    [0.5, 0.0],      # input 1: partition sum = 0.5
    [0.7, 0.3]       # input 2: partition sum = 1.0 
])

# Partitioned joint data (joint = prior * channel element-wise)
# Prior: [0.25, 0.5, 0.25]
# Part 0: prior (3x1) * ch_part_0 (3x2)
_JOINT_PART_0_EXPECTED = np.array([
    [0.15, 0.1],      # 0.25 * [0.6, 0.4]
    [0.15, 0.1],      # 0.5  * [0.3, 0.2]
    [0.0, 0.0]        # 0.25 * [0.0, 0.0]
])

# Part 1: prior (3x1) * ch_part_1 (3x2)
_JOINT_PART_1_EXPECTED = np.array([
    [0.0, 0.0],       # 0.25 * [0.0, 0.0]
    [0.25, 0.0],      # 0.5  * [0.5, 0.0]
    [0.175, 0.075]    # 0.25 * [0.7, 0.3]
])

# Partitioned hyper (outer and posterior from partitioned joint)
_OUTER_PART_EXPECTED = np.array([0.3, 0.2, 0.425, 0.075])

# Partitioned posteriors (part of hyper)
_POSTERIOR_PART_0_EXPECTED = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.0, 0.0]
])

_POSTERIOR_PART_1_EXPECTED = np.array([
    [0.0, 0.0],
    [0.58823529, 0.0],
    [0.41176471, 1.0]
])

# Partitioned strategies (from partitioned joint)
# Strategy uses max-margin rule: for each column, set entry with max value to 1
# (distributed equally if there are ties)
_STRATEGY_PART_0_EXPECTED = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.0, 0.0]
])

_STRATEGY_PART_1_EXPECTED = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
])


class TestJoint:
    @pytest.mark.parametrize(
        "ch_dist",
        [_CHANNEL_0, sp.csr_array(_CHANNEL_0)]
    )
    def test_joint(self, ch_dist):
        pi = ProbabDist(_PRIOR_0)
        ch = Channel(ch_dist)

        j = qif.joint(pi, ch)
        j_dist = j.dist
        if sp.issparse(j_dist): j_dist = j_dist.toarray()

        assert isinstance(j, Joint)
        assert j.is_complete
        np.testing.assert_allclose(j_dist, _JOINT_0)


    @pytest.mark.parametrize(
        "ch_dist",
        [[_CHANNEL_PART_0, _CHANNEL_PART_1],
         [sp.csr_array(_CHANNEL_PART_0), sp.csr_array(_CHANNEL_PART_1)]]
    )
    def test_joint_partitioned(self, ch_dist):
        pi = ProbabDist(_PRIOR_0)
        ch = Channel(ch_dist)

        j = qif.joint(pi, ch)

        assert isinstance(j, Joint)
        assert j.is_complete
        assert isinstance(j.dist, Sequence)
        assert len(j.dist) == 2

        j_part_0 = j.dist[0]
        j_part_1 = j.dist[1]

        if sp.issparse(j_part_0): j_part_0 = j_part_0.toarray()
        if sp.issparse(j_part_1): j_part_1 = j_part_1.toarray()

        np.testing.assert_allclose(j_part_0, _JOINT_PART_0_EXPECTED)
        np.testing.assert_allclose(j_part_1, _JOINT_PART_1_EXPECTED)
         

class TestHyper:
    @pytest.mark.parametrize(
        "ch_dist",
        [_CHANNEL_0, sp.csr_array(_CHANNEL_0)]
    )
    def test_hyper_from_pi_ch(self, ch_dist):
        pi = ProbabDist(_PRIOR_0)
        ch = Channel(ch_dist)

        h = qif.hyper(pi, ch)

        outer_dist = h.outer.dist
        post_dist = h.posteriors.dist
        if sp.issparse(post_dist): post_dist = post_dist.toarray()

        np.testing.assert_allclose(outer_dist, _OUTER_0)
        np.testing.assert_allclose(post_dist, _POSTERIOR_0)


    @pytest.mark.parametrize(
        "ch_dist",
        [[_CHANNEL_PART_0, _CHANNEL_PART_1],
         [sp.csr_array(_CHANNEL_PART_0), _CHANNEL_PART_1],
         [sp.csr_array(_CHANNEL_PART_0), sp.csr_array(_CHANNEL_PART_1)]]
    )
    def test_hyper_from_pi_ch_partitioned(self, ch_dist):
        pi = ProbabDist(_PRIOR_0)
        ch = Channel(ch_dist)

        h = qif.hyper(pi, ch)

        assert isinstance(h.outer, ProbabDist)
        assert isinstance(h.posteriors, StochMatrix)

        # Outer should be non-partitioned (flattened across partitions)
        outer_dist = h.outer.dist
        if sp.issparse(outer_dist): outer_dist = outer_dist.toarray()

        np.testing.assert_allclose(outer_dist, _OUTER_PART_EXPECTED)

        # Posterior should be partitioned
        post_dist = h.posteriors.dist
        assert isinstance(post_dist, Sequence)
        assert len(post_dist) == 2

        post_part_0 = post_dist[0]
        post_part_1 = post_dist[1]

        if sp.issparse(post_part_0): post_part_0 = post_part_0.toarray()
        if sp.issparse(post_part_1): post_part_1 = post_part_1.toarray()

        # Verify posterior values
        np.testing.assert_allclose(post_part_0, _POSTERIOR_PART_0_EXPECTED)
        np.testing.assert_allclose(post_part_1, _POSTERIOR_PART_1_EXPECTED)


    @pytest.mark.parametrize(
        "joint_dist",
        [[_JOINT_PART_0_EXPECTED, _JOINT_PART_1_EXPECTED],
         [sp.csr_array(_JOINT_PART_0_EXPECTED), _JOINT_PART_1_EXPECTED],
         [sp.csr_array(_JOINT_PART_0_EXPECTED),
          sp.csr_array(_JOINT_PART_1_EXPECTED)]]
    )
    def test_hyper_from_joint_partitioned(self, joint_dist):
        j = Joint(joint_dist)
        h = qif.hyper(j)

        assert isinstance(h.outer, ProbabDist)
        assert isinstance(h.posteriors, StochMatrix)

        # Outer should be non-partitioned
        outer_dist = h.outer.dist
        if sp.issparse(outer_dist): outer_dist = outer_dist.toarray()

        np.testing.assert_allclose(outer_dist, _OUTER_PART_EXPECTED)

        # Posterior should be partitioned
        post_dist = h.posteriors.dist
        assert isinstance(post_dist, Sequence)
        assert len(post_dist) == 2

        post_part_0 = post_dist[0]
        post_part_1 = post_dist[1]

        if sp.issparse(post_part_0): post_part_0 = post_part_0.toarray()
        if sp.issparse(post_part_1): post_part_1 = post_part_1.toarray()

        # Verify posterior values
        np.testing.assert_allclose(post_part_0, _POSTERIOR_PART_0_EXPECTED)
        np.testing.assert_allclose(post_part_1, _POSTERIOR_PART_1_EXPECTED)


    @pytest.mark.parametrize(
        "joint_dist",
        [_JOINT_0, sp.csr_array(_JOINT_0)]
    )
    def test_hyper_from_joint(self, joint_dist):
        j = Joint(joint_dist)
        h = qif.hyper(j)

        outer_dist = h.outer.dist
        post_dist = h.posteriors.dist
        if sp.issparse(post_dist): post_dist = post_dist.toarray()

        np.testing.assert_allclose(outer_dist, _OUTER_0)
        np.testing.assert_allclose(post_dist, _POSTERIOR_0)


class TestStrategy:
    @pytest.mark.parametrize(
        "pi_dist,expected",
        [(_PRIOR_0, _STRATEGY_PRIOR_0), (_PRIOR_1, _STRATEGY_PRIOR_1)]
    )
    def test_strategy_from_pi(self, pi_dist, expected):
        pi = ProbabDist(pi_dist)
        s = qif.strategy(pi)

        assert isinstance(s, Strategy)
        np.testing.assert_allclose(s.dist, expected)


    @pytest.mark.parametrize(
        "pi_dist,ch_dist,expected",
        [(_PRIOR_0, _CHANNEL_0, _STRATEGY_JOINT_0),
         (_PRIOR_2, _CHANNEL_1, _STRATEGY_JOINT_1),
         (_PRIOR_0, sp.csr_array(_CHANNEL_0), _STRATEGY_JOINT_0),
         (_PRIOR_2, sp.csr_array(_CHANNEL_1), _STRATEGY_JOINT_1)]
    )
    def test_strategy_from_pi_ch(self, pi_dist, ch_dist, expected):
        pi = ProbabDist(pi_dist)
        ch = Channel(ch_dist)

        s = qif.strategy(pi, ch)
        assert isinstance(s, Strategy)

        dist = s.dist.toarray() if sp.issparse(s.dist) else s.dist
        np.testing.assert_allclose(dist, expected)

        
    @pytest.mark.parametrize("joint_dist,expected", [
        (_JOINT_0, _STRATEGY_JOINT_0),
        (_JOINT_1, _STRATEGY_JOINT_1),
        (sp.csr_array(_JOINT_0), _STRATEGY_JOINT_0),
        (sp.csr_array(_JOINT_1), _STRATEGY_JOINT_1),
    ])
    def test_strategy_from_joint(self, joint_dist, expected):
        j = Joint(joint_dist)

        s = qif.strategy(j)
        assert isinstance(s, Strategy)

        dist = s.dist.toarray() if sp.issparse(s.dist) else s.dist
        np.testing.assert_allclose(dist, expected)


    @pytest.mark.parametrize(
        "joint_dist",
        [[_JOINT_PART_0_EXPECTED, _JOINT_PART_1_EXPECTED],
         [sp.csr_array(_JOINT_PART_0_EXPECTED), _JOINT_PART_1_EXPECTED],
         [sp.csr_array(_JOINT_PART_0_EXPECTED),
          sp.csr_array(_JOINT_PART_1_EXPECTED)]]
    )
    def test_strategy_from_joint_partitioned(self, joint_dist):
        j = Joint(joint_dist)
        s = qif.strategy(j)

        assert isinstance(s, Strategy)

        # Strategy should be partitioned
        assert isinstance(s.dist, Sequence)
        assert len(s.dist) == 2

        s_part_0 = s.dist[0]
        s_part_1 = s.dist[1]

        if sp.issparse(s_part_0): s_part_0 = s_part_0.toarray()
        if sp.issparse(s_part_1): s_part_1 = s_part_1.toarray()

        np.testing.assert_allclose(s_part_0, _STRATEGY_PART_0_EXPECTED)
        np.testing.assert_allclose(s_part_1, _STRATEGY_PART_1_EXPECTED)
