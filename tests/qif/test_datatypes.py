import numpy as np
import pytest
import scipy.sparse as sp

from qif_micro.qif.datatypes import Channel, ProbabDist, Joint

class TestChannel:
    def test_channel_complete(self):
        ch = Channel(np.array([[0.5, 0.5], [1.0, 0.0]]))
        assert ch.is_complete

        ch = Channel(sp.csr_array(ch.dist))
        assert ch.is_complete


    def test_channel_complete_partitioned(self):
        part0 = np.array([[0.2, 0.2], [0.5, 0.0]])
        part1 = np.array([[0.6], [0.5]])

        ch = Channel([part0, part1])
        assert ch.is_complete

        part0 = sp.csr_array(part0)
        ch = Channel([part0, part1])
        assert ch.is_complete

        part1 = sp.csr_array(part1)
        ch = Channel([part0, part1])
        assert ch.is_complete

        
    def test_channel_slice(self):
        ch = Channel(np.array([[0.2, 0.3], [0.5, 0.0]]))
        assert not ch.is_complete

        ch = Channel(sp.csr_array(ch.dist))
        assert not ch.is_complete


    def test_channel_slice_partitioned(self):
        part0 = np.array([[0.2, 0.1], [0.3, 0.0]])
        part1 = np.array([[0.6], [0.7]])

        ch = Channel([part0, part1])
        assert not ch.is_complete

        part0 = sp.csr_array(part0)
        ch = Channel([part0, part1])
        assert not ch.is_complete

        part1 = sp.csr_array(part1)
        ch = Channel([part0, part1])
        assert not ch.is_complete


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
        pd = ProbabDist(np.array([0.25, 0.5, 0.25]))
        assert pd.is_complete

        pd = ProbabDist(sp.csr_array([0.25, 0.5, 0.25]))
        assert pd.is_complete


    def test_probab_dist_slice(self):
        pd = ProbabDist(np.array([0.2, 0.3]))
        assert not pd.is_complete

        pd = ProbabDist(sp.csr_array([0.2, 0.3]))
        assert not pd.is_complete


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
        j = Joint(np.array([[0.25, 0.25], [0.25, 0.25]]))
        assert j.is_complete

        j = Joint(sp.csr_array([[0.25, 0.25], [0.25, 0.25]]))
        assert j.is_complete


    def test_joint_complete_partitioned(self):
        part0 = np.array([[0.2, 0.2], [0.25, 0.0]])
        part1 = np.array([[0.1], [0.25]])

        j = Joint([part0, part1])
        assert j.is_complete

        part0 = sp.csr_array(part0)
        j = Joint([part0, part1])
        assert j.is_complete

        part1 = sp.csr_array(part1)
        j = Joint([part0, part1])
        assert j.is_complete


    def test_joint_slice_dense(self):
        j = Joint(np.array([[0.2, 0.1], [0.1, 0.0]]))
        assert not j.is_complete

        j = Joint(sp.csr_array([[0.2, 0.1], [0.1, 0.0]]))
        assert not j.is_complete


    def test_joint_slice_partitioned(self):
        part0 = np.array([[0.2, 0.0], [0.25, 0.0]])
        part1 = np.array([[0.1], [0.25]])

        j = Joint([part0, part1])
        assert not j.is_complete

        part0 = sp.csr_array(part0)
        j = Joint([part0, part1])
        assert not j.is_complete

        part1 = sp.csr_array(part1)
        j = Joint([part0, part1])
        assert not j.is_complete


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
