import numpy as np
import scipy.sparse as sp
import pytest

from qif_micro import qif
from qif_micro.qif.datatypes import Channel


CHANNEL_IDENTITY = np.array([[1, 0], [0, 1]])
CHANNEL_NI = np.array([[1], [1]])

CHANNEL_PART_0 = np.array([[1/2, 1/4], [0, 1/6]])
CHANNEL_PART_1 = np.array([[0, 1/4], [2/3, 1/6]])
CHANNEL_0 = np.hstack([CHANNEL_PART_0, CHANNEL_PART_1])

CHANNEL_1 = np.array([
    [2/3, 1/6, 1/6],
    [2/3, 1/3, 0]
])

class TestParallel:
    # ========================================================================
    # Valid input tests (complementing docstring)
    # ========================================================================
    @pytest.mark.parametrize(
        "lhs_dist,rhs_dist",
        [(CHANNEL_0, CHANNEL_1),
         (sp.csr_array(CHANNEL_0), CHANNEL_1),
         (CHANNEL_0, sp.csr_array(CHANNEL_1)),
         (sp.csr_array(CHANNEL_0), sp.csr_array(CHANNEL_1))]
    )
    def test_parallel_no_opt_memory(self, lhs_dist, rhs_dist):
        lhs = Channel(lhs_dist)
        rhs = Channel(rhs_dist)

        # Should not have all-zero columns:
        expected = np.array([
            [1/3, 1/12, 1/12, 1/6, 1/24, 1/24,   0,   0, 1/6, 1/24, 1/24],
            [0,      0,    0, 1/9, 1/18,    0, 4/9, 2/9, 1/9, 1/18, 0]
        ])

        ch = qif.compose.parallel(lhs, rhs, opt_memory=False)
        assert isinstance(ch, Channel)
        assert ch.dist.shape == (2, expected.shape[1])
        assert ch.is_complete

        dist = ch.dist.toarray() if sp.issparse(ch.dist) else ch.dist
        np.testing.assert_allclose(dist, expected)


    @pytest.mark.parametrize(
        "lhs_dist,rhs_dist",
        [([CHANNEL_PART_0, CHANNEL_PART_1], CHANNEL_1),
         ([sp.csr_array(CHANNEL_PART_0), CHANNEL_PART_1], CHANNEL_1)]
    )
    def test_parallel_no_opt_memory_partitioned(self, lhs_dist, rhs_dist):
        lhs = Channel(lhs_dist)
        rhs = Channel(rhs_dist)

        # Should not have all-zero columns:
        expected_part_0 = np.array([
            [1/3, 1/12, 1/12, 1/6, 1/24, 1/24],
            [0,      0,    0, 1/9, 1/18,    0]
        ])

        expected_part_1 = np.array([
            [  0,   0, 1/6, 1/24, 1/24],
            [4/9, 2/9, 1/9, 1/18, 0]
        ])

        ch = qif.compose.parallel(lhs, rhs, opt_memory=False)
        assert isinstance(ch, Channel)
        assert len(ch.dist) == 2
        assert ch.dist[0].shape == (2, expected_part_0.shape[1])
        assert ch.dist[1].shape == (2, expected_part_1.shape[1])
        assert ch.is_complete

        is_sparse = sp.issparse(ch.dist[0])
        if is_sparse: assert sp.issparse(ch.dist[1]) # Both are sparse

        dist_0 = ch.dist[0].toarray() if is_sparse else ch.dist[0]
        dist_1 = ch.dist[1].toarray() if is_sparse else ch.dist[1]

        np.testing.assert_allclose(dist_0, expected_part_0)
        np.testing.assert_allclose(dist_1, expected_part_1)


    @pytest.mark.parametrize(
        "lhs_dist,rhs_dist",
        [(sp.csr_array(CHANNEL_0), CHANNEL_1),
         (CHANNEL_0, sp.csr_array(CHANNEL_1)),
         (sp.csr_array(CHANNEL_0), sp.csr_array(CHANNEL_1))]
    )
    def test_parallel_opt_memory_left(self, lhs_dist, rhs_dist):
        lhs = Channel(lhs_dist)
        rhs = Channel(rhs_dist)

        # Should not have all-zero columns:
        expected_opt = np.array([[1/2, 0], [0, 2/3]])
        expected_rest = np.array([
            [1/6, 1/24, 1/24, 1/6, 1/24, 1/24],
            [1/9, 1/18,    0, 1/9, 1/18,    0]
        ])

        expected_cols = np.array([
            [0, -1], # Reduced from lhs
            [2, -1], # Reduced from lhs
            [1, 0],
            [1, 1],
            [1, 2],
            [3, 0],
            [3, 1],
            [3, 2]
        ])

        ch, cols = qif.compose.parallel(
            lhs, rhs, opt_memory=True, return_cols=True
        )

        assert isinstance(ch, Channel)
        assert ch.dist[0].shape == (2, expected_opt.shape[1]) # Opt columns
        assert ch.dist[1].shape == (2, expected_rest.shape[1]) # Rest
        assert ch.is_complete

        dist = sp.hstack(ch.dist).toarray()
        expected = np.hstack([expected_opt, expected_rest])

        np.testing.assert_array_equal(cols, expected_cols)
        np.testing.assert_allclose(dist, expected)


    def test_parallel_opt_memory_left_partitioned(self):
        lhs = Channel([sp.csr_array(CHANNEL_PART_0), CHANNEL_PART_1])
        rhs = Channel(CHANNEL_1)

        # Should not have all-zero columns:
        expected_opt = np.array([[1/2, 0], [0, 2/3]])
        expected_part_0 = np.array([
            [1/6, 1/24, 1/24],
            [1/9, 1/18,    0]
        ])

        expected_part_1 = np.array([
            [1/6, 1/24, 1/24],
            [1/9, 1/18,    0]
        ])

        expected_cols = np.array([
            [0, -1], # Reduced from lhs
            [2, -1], # Reduced from lhs
            [1, 0],
            [1, 1],
            [1, 2],
            [3, 0],
            [3, 1],
            [3, 2]
        ])

        ch, cols = qif.compose.parallel(
            lhs, rhs, opt_memory=True, return_cols=True
        )

        assert isinstance(ch, Channel)
        assert len(ch.dist) == 3
        assert ch.dist[0].shape == (2, expected_opt.shape[1]) # Opt columns
        assert ch.dist[1].shape == (2, expected_part_0.shape[1])
        assert ch.dist[2].shape == (2, expected_part_1.shape[1])
        assert ch.is_complete

        is_sparse = sp.issparse(ch.dist[0])
        if is_sparse: assert sp.issparse(ch.dist[1]) # All are sparse
        if is_sparse: assert sp.issparse(ch.dist[2]) # All are sparse

        dist_0 = ch.dist[0].toarray() if is_sparse else ch.dist[0]
        dist_1 = ch.dist[1].toarray() if is_sparse else ch.dist[1]
        dist_2 = ch.dist[2].toarray() if is_sparse else ch.dist[2]

        np.testing.assert_allclose(dist_0, expected_opt)
        np.testing.assert_allclose(dist_1, expected_part_0)
        np.testing.assert_allclose(dist_2, expected_part_1)


    @pytest.mark.parametrize(
        "lhs_dist,rhs_dist",
        [(sp.csr_array(CHANNEL_1), CHANNEL_IDENTITY),
         (CHANNEL_1, sp.csr_array(CHANNEL_IDENTITY)),
         (sp.csr_array(CHANNEL_1), sp.csr_array(CHANNEL_IDENTITY))]
    )
    def test_parallel_opt_memory_both(self, lhs_dist, rhs_dist):
        lhs = Channel(lhs_dist)
        rhs = Channel(rhs_dist)

        # Should not have all-zero columns:
        expected_opt = np.array([[1/6, 0], [0, 1]])
        expected_rest = np.array([[2/3, 1/6], [0, 0]])
        expected_cols = np.array([
            [2, -1], # Reduced from lhs
            [-1, 1], # Reduce from rhs
            [0, 0],
            [1, 0]
        ])

        ch, cols = qif.compose.parallel(
            lhs, rhs, opt_memory=True, return_cols=True
        )

        dist_opt = sp.hstack(ch.dist[:2])
        dist_rest = ch.dist[-1]

        assert isinstance(ch, Channel)
        assert dist_opt.shape == (2, expected_opt.shape[1]) # Opt columns
        assert dist_rest.shape == (2, expected_rest.shape[1]) # Rest
        assert ch.is_complete

        dist = sp.hstack(ch.dist).toarray()
        expected = np.hstack([expected_opt, expected_rest])

        np.testing.assert_array_equal(cols, expected_cols)
        np.testing.assert_allclose(dist, expected)


    @pytest.mark.parametrize(
        "lhs_dist,rhs_dist",
        [(CHANNEL_IDENTITY, CHANNEL_IDENTITY), (CHANNEL_IDENTITY, CHANNEL_0)]
    )
    def test_parallel_identity(self, lhs_dist, rhs_dist):
        lhs = Channel(lhs_dist)
        rhs = Channel(rhs_dist)

        row_0 = np.vstack([rhs_dist[0, :], np.repeat(0, rhs_dist.shape[1])])
        row_1 = np.vstack([np.repeat(0, rhs_dist.shape[1]), rhs_dist[1, :]])
        naive_matrix = np.hstack([row_0, row_1])
        keep = naive_matrix.any(axis=0) # Drop all-zero columns
        expected = naive_matrix[:, keep]

        ch = qif.compose.parallel(lhs, rhs, opt_memory=False)
        assert isinstance(ch, Channel)
        assert ch.dist.shape == (2, expected.shape[1])
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist, expected)


    @pytest.mark.parametrize(
        "lhs_dist,rhs_dist",
        [(CHANNEL_NI, CHANNEL_0), (CHANNEL_0, CHANNEL_NI)]
    )
    def test_parallel_no_intereference(self, lhs_dist, rhs_dist):
        lhs = Channel(lhs_dist)
        rhs = Channel(rhs_dist)

        ch = qif.compose.parallel(lhs, rhs, opt_memory=False)
        assert isinstance(ch, Channel)
        assert ch.dist.shape == (2, 4)
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist, CHANNEL_0)


    # ========================================================================
    # Invalid inputs - raises error
    # ========================================================================
    def test_parallel_row_count_mismatch(self):
        lhs = Channel(np.array([[1/2, 1/2], [1/2, 1/2]]))
        rhs = Channel(np.array([[1/3, 1/3, 1/3]]))

        with pytest.raises(ValueError, match="Number of rows"):
            qif.compose.parallel(lhs, rhs)
