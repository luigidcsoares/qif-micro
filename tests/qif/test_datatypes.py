import numpy as np
import pytest
import scipy.sparse as sp

from qif_micro.qif.datatypes import Channel, ProbabDist, Joint, Hyper, StochMatrix

class TestChannel:
    def test_channel_complete(self):
        dist = np.array([[0.5, 0.5], [1.0, 0.0]])

        ch = Channel(dist)
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist, dist)

        ch = Channel(sp.csr_array(dist))
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist.toarray(), dist)


    def test_channel_complete_partitioned(self):
        part0 = np.array([[0.2, 0.2], [0.5, 0.0]])
        part1 = np.array([[0.6], [0.5]])

        ch = Channel([part0, part1])
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist[0], part0)
        np.testing.assert_allclose(ch.dist[1], part1)

        part0 = sp.csr_array(part0)
        ch = Channel([part0, part1])
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(ch.dist[1], part1)

        part1 = sp.csr_array(part1)
        ch = Channel([part0, part1])
        assert ch.is_complete
        np.testing.assert_allclose(ch.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(ch.dist[1].toarray(), part1.toarray())

        
    def test_channel_slice(self):
        dist = np.array([[0.2, 0.3], [0.5, 0.0]])
        ch = Channel(dist)
        assert not ch.is_complete
        np.testing.assert_allclose(ch.dist, dist)

        ch = Channel(sp.csr_array(dist))
        assert not ch.is_complete
        np.testing.assert_allclose(ch.dist.toarray(), dist)


    def test_channel_slice_partitioned(self):
        part0 = np.array([[0.2, 0.1], [0.3, 0.0]])
        part1 = np.array([[0.6], [0.7]])

        ch = Channel([part0, part1])
        assert not ch.is_complete
        np.testing.assert_allclose(ch.dist[0], part0)
        np.testing.assert_allclose(ch.dist[1], part1)

        part0 = sp.csr_array(part0)
        ch = Channel([part0, part1])
        assert not ch.is_complete
        np.testing.assert_allclose(ch.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(ch.dist[1], part1)

        part1 = sp.csr_array(part1)
        ch = Channel([part0, part1])
        assert not ch.is_complete
        np.testing.assert_allclose(ch.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(ch.dist[1].toarray(), part1.toarray())


    def test_channel_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Channel(np.array([0.5, 0.5]))

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Channel(sp.csr_array([0.5, 0.5]))

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Channel([np.array([0.5]), sp.csr_array([0.5])])


    def test_channel_rejects_negative(self):
        with pytest.raises(ValueError, match="Negative"):
            Channel(np.array([[-0.1, 1.1], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="Negative"):
            Channel(sp.csr_array([[-0.1, 1.1], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="Negative"):
            Channel([sp.csr_array([[-0.1], [0.5]]), np.array([[1.1], [0.5]])])


    def test_channel_rejects_row_sum_over_one(self):
        with pytest.raises(ValueError, match="exceeds 1"):
            Channel(np.array([[1.5, 0.5], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="exceeds 1"):
            Channel(sp.csr_array([[1.5, 0.5], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="exceeds 1"):
            Channel([sp.csr_array([[1.5], [0.5]]), np.array([[0.5], [0.5]])])


class TestProbabDist:
    def test_probab_dist_complete(self):
        dist = np.array([0.25, 0.5, 0.25])

        pd = ProbabDist(dist)
        assert pd.is_complete
        np.testing.assert_allclose(pd.dist, dist)


    def test_probab_dist_slice(self):
        dist = np.array([0.2, 0.3])

        pd = ProbabDist(dist)
        assert not pd.is_complete
        np.testing.assert_allclose(pd.dist, dist)


    def test_probab_dist_must_be_ndarray(self):
        with pytest.raises(ValueError, match="must be ndarray"):
            ProbabDist([[0.5, 0.5]])

        with pytest.raises(ValueError, match="must be ndarray"):
            ProbabDist(sp.csr_array([[0.5, 0.5]]))


    def test_probab_dist_rejects_2d(self):
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            ProbabDist(np.array([[0.5, 0.5]]))


    def test_probab_dist_rejects_negative(self):
        with pytest.raises(ValueError, match="Negative"):
            ProbabDist(np.array([-0.1, 1.1]))


    def test_probab_dist_rejects_sum_over_one(self):
        with pytest.raises(ValueError, match="exceeds 1"):
            ProbabDist(np.array([0.6, 0.5]))


class TestJoint:
    def test_joint_complete(self):
        dist = np.array([[0.25, 0.25], [0.25, 0.25]])

        j = Joint(dist)
        assert j.is_complete
        np.testing.assert_allclose(j.dist, dist)

        j = Joint(sp.csr_array(dist))
        assert j.is_complete
        np.testing.assert_allclose(j.dist.toarray(), dist)


    def test_joint_complete_partitioned(self):
        part0 = np.array([[0.2, 0.2], [0.25, 0.0]])
        part1 = np.array([[0.1], [0.25]])

        j = Joint([part0, part1])
        assert j.is_complete
        np.testing.assert_allclose(j.dist[0], part0)
        np.testing.assert_allclose(j.dist[1], part1)

        part0 = sp.csr_array(part0)
        j = Joint([part0, part1])
        assert j.is_complete
        np.testing.assert_allclose(j.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(j.dist[1], part1)

        part1 = sp.csr_array(part1)
        j = Joint([part0, part1])
        assert j.is_complete
        np.testing.assert_allclose(j.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(j.dist[1].toarray(), part1.toarray())


    def test_joint_slice_dense(self):
        dist = np.array([[0.2, 0.1], [0.1, 0.0]])

        j = Joint(dist)
        assert not j.is_complete
        np.testing.assert_allclose(j.dist, dist)

        j = Joint(sp.csr_array(dist))
        assert not j.is_complete
        np.testing.assert_allclose(j.dist.toarray(), dist)


    def test_joint_slice_partitioned(self):
        part0 = np.array([[0.2, 0.0], [0.25, 0.0]])
        part1 = np.array([[0.1], [0.25]])

        j = Joint([part0, part1])
        assert not j.is_complete
        np.testing.assert_allclose(j.dist[0], part0)
        np.testing.assert_allclose(j.dist[1], part1)

        j = Joint([part0, part1])
        assert not j.is_complete
        np.testing.assert_allclose(j.dist[0], part0)
        np.testing.assert_allclose(j.dist[1], part1)

        part0 = sp.csr_array(part0)
        j = Joint([part0, part1])
        assert not j.is_complete
        np.testing.assert_allclose(j.dist[0].toarray(), part0.toarray())
        np.testing.assert_allclose(j.dist[1], part1)


    def test_joint_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Joint(np.array([0.25, 0.25]))

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Joint(sp.csr_array([0.25, 0.25]))

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Joint([np.array([0.25]), sp.csr_array([0.25])])


    def test_joint_rejects_negative(self):
        with pytest.raises(ValueError, match="Negative"):
            Joint(np.array([[-0.1, 1.1], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="Negative"):
            Joint(sp.csr_array([[-0.1, 1.1], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="Negative"):
            Joint([sp.csr_array([[-0.1], [0.5]]), np.array([[1.1], [0.5]])])


    def test_joint_rejects_sum_over_one(self):
        with pytest.raises(ValueError, match="exceeds 1"):
            Joint(np.array([[0.6, 0.6], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="exceeds 1"):
            Joint(sp.csr_array([[0.6, 0.6], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="exceeds 1"):
            Joint([sp.csr_array([[0.6], [0.5]]), np.array([[0.6], [0.5]])])


class TestHyper:
    def test_hyper_complete(self):
        outer = ProbabDist(np.array([0.4, 0.6]))
        # Posterior: (n_inputs=2, n_outputs=2), columns sum to 1.0
        posterior = StochMatrix(
            np.array([[2/3, 1/3], [1/3, 2/3]]), dist_orient=0
        )

        h = Hyper(outer, posterior)
        assert h.outer.is_complete
        assert h.posteriors.is_complete
        np.testing.assert_allclose(h.outer.dist, outer.dist)
        np.testing.assert_allclose(h.posteriors.dist, posterior.dist)

        posterior_sparse = StochMatrix(sp.csr_array(posterior.dist), dist_orient=0)
        h = Hyper(outer, posterior_sparse)
        assert h.outer.is_complete
        assert h.posteriors.is_complete
        np.testing.assert_allclose(h.outer.dist, outer.dist)
        np.testing.assert_allclose(
            h.posteriors.dist.toarray(), posterior.dist
        )


    def test_hyper_complete_partitioned(self):
        outer = ProbabDist(np.array([0.4, 0.6]))
        # Partitioned by columns: part0 has output 0, part1 has output 1
        part0 = np.array([[2/3], [1/3]])     # 2 inputs, 1 output
        part1 = np.array([[1/3], [2/3]])     # 2 inputs, 1 output

        posterior = StochMatrix([part0, part1], dist_orient=0)
        h = Hyper(outer, posterior)
        assert h.outer.is_complete
        assert h.posteriors.is_complete
        np.testing.assert_allclose(h.posteriors.dist[0], part0)
        np.testing.assert_allclose(h.posteriors.dist[1], part1)

        part0_sparse = sp.csr_array(part0)
        posterior_sparse = StochMatrix([part0_sparse, part1], dist_orient=0)
        h = Hyper(outer, posterior_sparse)
        assert h.outer.is_complete
        assert h.posteriors.is_complete
        np.testing.assert_allclose(
            h.posteriors.dist[0].toarray(), part0_sparse.toarray()
        )
        np.testing.assert_allclose(h.posteriors.dist[1], part1)

        part1_sparse = sp.csr_array(part1)
        posterior_both_sparse = StochMatrix(
            [part0_sparse, part1_sparse], dist_orient=0
        )
        h = Hyper(outer, posterior_both_sparse)
        assert h.outer.is_complete
        assert h.posteriors.is_complete
        np.testing.assert_allclose(
            h.posteriors.dist[0].toarray(), part0_sparse.toarray()
        )
        np.testing.assert_allclose(
            h.posteriors.dist[1].toarray(), part1_sparse.toarray()
        )


    def test_hyper_rejects_non_probab_dist_outer(self):
        with pytest.raises(TypeError, match="must be a ProbabDist"):
            posterior = StochMatrix(
                np.array([[2/3, 1/3], [1/3, 2/3]]), dist_orient=0
            )
            Hyper(np.array([0.4, 0.6]), posterior)


    def test_hyper_rejects_non_stoch_matrix_posterior(self):
        with pytest.raises(TypeError, match="must be a StochMatrix"):
            outer = ProbabDist(np.array([0.4, 0.6]))
            Hyper(outer, np.array([[0.4, 0.6], [0.6, 0.4]]))


    def test_hyper_rejects_wrong_posterior_orient(self):
        with pytest.raises(ValueError, match="dist_orient=0"):
            outer = ProbabDist(np.array([0.4, 0.6]))
            # Create StochMatrix with dist_orient=1 (channel, not posterior)
            posterior = StochMatrix(
                np.array([[2/3, 1/3], [1/3, 2/3]]), dist_orient=1
            )
            Hyper(outer, posterior)


    def test_hyper_accepts_incomplete_posterior(self):
        # Columns sum to < 1.0 (incomplete, but valid)
        outer = ProbabDist(np.array([0.4, 0.6]))
        posterior = StochMatrix(
            np.array([[0.4, 0.4], [0.4, 0.4]]), dist_orient=0
        )
        h = Hyper(outer, posterior)
        assert not h.posteriors.is_complete
        assert h.posteriors is not None

        posterior_sparse = StochMatrix(
            [sp.csr_array([[0.4, 0.4]]), np.array([[0.4, 0.4]])],
            dist_orient=0
        )
        h = Hyper(outer, posterior_sparse)
        assert not h.posteriors.is_complete
        assert h.posteriors is not None


    def test_hyper_rejects_posterior_col_sum_over_one(self):
        with pytest.raises(ValueError, match="Sum of columns exceeds 1"):
            outer = ProbabDist(np.array([0.4, 0.6]))
            posterior = StochMatrix(
                np.array([[0.6, 0.6], [0.5, 0.5]]), dist_orient=0
            )
            Hyper(outer, posterior)

        # Partitioned: part0 and part1 each have 1 column (1 output)
        # part0 column sum = 1.1, part1 column sum = 0.9
        with pytest.raises(ValueError, match="Sum of columns exceeds 1"):
            outer = ProbabDist(np.array([0.4, 0.6]))
            posterior = StochMatrix(
                [sp.csr_array([[0.6], [0.5]]), np.array([[0.4], [0.5]])],
                dist_orient=0
            )
            Hyper(outer, posterior)

