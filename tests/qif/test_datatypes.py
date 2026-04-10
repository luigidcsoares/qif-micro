import numpy as np
import pytest
import scipy.sparse as sp

from qif_micro.qif.datatypes import Channel, ProbabDist, Joint

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

        pd = ProbabDist(sp.csr_array(dist))
        assert pd.is_complete
        np.testing.assert_allclose(pd.dist.toarray().ravel(), dist)


    def test_probab_dist_slice(self):
        dist = np.array([0.2, 0.3])

        pd = ProbabDist(dist)
        assert not pd.is_complete
        np.testing.assert_allclose(pd.dist, dist)

        pd = ProbabDist(sp.csr_array(dist))
        assert not pd.is_complete
        np.testing.assert_allclose(pd.dist.toarray().ravel(), dist)


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
