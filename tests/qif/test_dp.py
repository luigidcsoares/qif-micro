import math
import pytest

import numpy as np

from qif_micro import qif
from qif_micro.qif.datatypes import Channel

class TestGeometric:
    # ========================================================================
    # Valid input tests (complementing docstring)
    # ========================================================================
    def test_geometric_slice_not_contiguous(self):
        input_domain = [0, 1]
        output_domain = [0, 1, 3]
        eps = -math.log(0.5) # alpha = 0.5

        ch = qif.dp.geometric(
            eps,
            input_domain,
            output_domain,
        )

        assert isinstance(ch, Channel)
        assert not ch.is_complete
        assert ch.dist.shape == (2, 3)

        expected = np.array([
            [0.66666667, 0.16666667, 0.08333333],
            [0.33333333, 0.33333333, 0.16666667]
        ])


    def test_geometric_duplicate_domain_values(self):
        unique_domain = [0, 1, 2]
        duplicate_domain = [2, 0, 1]
        eps = -math.log(0.5) # alpha = 1/2

        ch_unique = qif.dp.geometric(eps, unique_domain)
        ch_duplicate = qif.dp.geometric(eps, duplicate_domain)

        assert isinstance(ch_duplicate, Channel)
        assert ch_duplicate.is_complete
        assert ch_duplicate.dist.shape == (3, 3)

        np.testing.assert_allclose(ch_duplicate.dist, ch_unique.dist)


    def test_geometric_unsorted_domain(self):
        sorted_domain = [0, 1, 2]
        unsorted_domain = [2, 0, 1]
        eps = -math.log(0.5) # alpha = 1/2

        ch_sorted = qif.dp.geometric(eps, sorted_domain)
        ch_unsorted = qif.dp.geometric(eps, unsorted_domain)

        assert isinstance(ch_unsorted, Channel)
        assert ch_unsorted.is_complete
        assert ch_unsorted.dist.shape == (3, 3)

        np.testing.assert_allclose(ch_unsorted.dist, ch_sorted.dist)


    def test_geometric_zero_eps(self):
        input_domain = [0, 1, 2]
        eps = 0.0
        ch = qif.dp.geometric(eps, input_domain)

        assert isinstance(ch, Channel)
        assert ch.is_complete
        assert ch.dist.shape == (3, 3)

        # eps = 0 means infinite noise, which means that
        # half of the weight is in the first column and
        # the other half is in the last column, and every
        # other column is full of zeros.
        assert np.all(ch.dist[:, 0] == 0.5)
        assert np.all(ch.dist[:, 2] == 0.5)
        assert np.all(ch.dist[:, 1:2] == 0.0)

        
    # ========================================================================
    # Invalid epsilon tests
    # ========================================================================
    def test_geometric_negative_eps(self):
        with pytest.raises(ValueError, match="Privacy param .* must be >= 0!"):
            qif.dp.geometric(-0.5, [0, 1, 2])


    def test_geometric_inf_eps(self):
        with pytest.raises(ValueError, match="Privacy param .* must be finite!"):
            qif.dp.geometric(np.inf, [0, 1, 2])


    # ========================================================================
    # Invalid domain tests
    # ========================================================================
    def test_geometric_empty_input_domain(self):
        with pytest.raises(ValueError, match=".*input_domain.* cannot be empty!"):
            qif.dp.geometric(0.5, [])


    def test_geometric_empty_output_domain(self):
        with pytest.raises(ValueError, match=".*output_domain.* cannot be empty!"):
            qif.dp.geometric(0.5, [0, 1, 2], [])


    def test_geometric_non_integer_input_domain(self):
        with pytest.raises(ValueError, match=".*input_domain.* must contain only integers!"):
            qif.dp.geometric(0.5, [0.5, 1.5, 2.5])

        with pytest.raises(ValueError, match=".*input_domain.* must contain only integers!"):
            qif.dp.geometric(0.5, ["a", "b"])

            
    def test_geometric_non_integer_output_domain(self):
        with pytest.raises(ValueError, match=".*output_domain.* must contain only integers!"):
            qif.dp.geometric(0.5, [0, 1, 2], [0.5, 1.5, 2.5])

        with pytest.raises(ValueError, match=".*output_domain.* must contain only integers!"):
            qif.dp.geometric(0.5, [0, 1, 2], ["a", "b"])


    def test_geometric_output_not_superset(self):
        with pytest.raises(ValueError, match="Full channel: output must be a superset of input!"):
            qif.dp.geometric(0.5, [0, 1], [0, 2])


    def test_geometric_invalid_bounds(self):
        input_domain = [0, 1, 2]

        with pytest.raises(ValueError, match=r".*domain_max.* must be >=.*output_domain"):
            qif.dp.geometric(0.5, input_domain, domain_max=1)

        with pytest.raises(ValueError, match=r".*domain_min.* must be <=.*output_domain"):
            qif.dp.geometric(0.5, input_domain, domain_min=1)


    def test_geometric_strict_bounds(self):
        input_domain = [0, 1]
        output_domain = [0]

        with pytest.raises(ValueError, match=r".*domain_max.* must be > .*domain_min"):
            qif.dp.geometric(0.5, input_domain, output_domain)

        with pytest.raises(ValueError, match=r".*domain_max.* must be > .*domain_min"):
            qif.dp.geometric(0.5, input_domain, domain_min=1)


class TestRandomResponse:
    # ========================================================================
    # Valid input tests (complementing docstring)
    # ========================================================================
    def test_random_response_zero_eps(self):
        input_domain = [0, 1, 2]
        eps = 0.0
        ch = qif.dp.random_response(eps, input_domain)

        assert isinstance(ch, Channel)
        assert ch.is_complete
        assert ch.dist.shape == (3, 3)

        # eps = 0 means infinite noise, which means that
        # all cells have exactly the same probability: 1 / n
        assert np.all(ch.dist == 1 / len(input_domain))


    def test_random_response_duplicate_domain_values(self):
        unique_domain = [0, 1, 2]
        duplicate_domain = [2, 0, 1]
        eps = math.log(3) # alpha = 1/2

        ch_unique = qif.dp.random_response(eps, unique_domain)
        ch_duplicate = qif.dp.random_response(eps, duplicate_domain)

        assert isinstance(ch_duplicate, Channel)
        assert ch_duplicate.is_complete
        assert ch_duplicate.dist.shape == (3, 3)

        np.testing.assert_allclose(ch_duplicate.dist, ch_unique.dist)


    def test_random_response_unsorted_domain(self):
        sorted_domain = [0, 1, 2]
        unsorted_domain = [2, 0, 1]
        eps = math.log(3) # alpha = 1/2

        ch_sorted = qif.dp.random_response(eps, sorted_domain)
        ch_unsorted = qif.dp.random_response(eps, unsorted_domain)

        assert isinstance(ch_unsorted, Channel)
        assert ch_unsorted.is_complete
        assert ch_unsorted.dist.shape == (3, 3)

        np.testing.assert_allclose(ch_unsorted.dist, ch_sorted.dist)


    # ========================================================================
    # Invalid epsilon tests
    # ========================================================================
    def test_random_response_negative_eps(self):
        with pytest.raises(ValueError, match="Privacy param .* must be >= 0!"):
            qif.dp.random_response(-0.5, [0, 1, 2])


    def test_random_response_inf_eps(self):
        with pytest.raises(ValueError, match="Privacy param .* must be finite!"):
            qif.dp.random_response(np.inf, [0, 1, 2])


    # ========================================================================
    # Invalid domain tests
    # ========================================================================
    def test_random_response_empty_input_domain(self):
        with pytest.raises(ValueError, match=".*input_domain.* cannot be empty!"):
            qif.dp.random_response(0.5, [])


    def test_random_response_empty_output_domain(self):
        with pytest.raises(ValueError, match=".*output_domain.* cannot be empty!"):
            qif.dp.random_response(0.5, [0, 1, 2], [])


    def test_random_response_invalid_domain_size(self):
        with pytest.raises(ValueError, match=r".*domain_size.* must be >= 2"):
            qif.dp.random_response(0.5, [0])

        with pytest.raises(ValueError, match=r".*domain_size.* must be >= 2"):
            qif.dp.random_response(0.5, [0, 1], domain_size=1)


    def test_random_response_exceed_domain_size(self):
        with pytest.raises(ValueError, match=r".*input_domain.* more values.*domain_size"):
            qif.dp.random_response(0.5, [0, 1, 2], domain_size=2)

        with pytest.raises(ValueError, match=r".*output_domain.* more values.*domain_size"):
            qif.dp.random_response(0.5, [0, 1], [0, 1, 2], domain_size=2)


    def test_random_response_output_not_superset(self):
        with pytest.raises(ValueError, match="Full channel: output must be a superset of input!"):
            qif.dp.random_response(0.5, [0, 1], [0, 2])

