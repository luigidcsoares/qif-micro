from numpy import ndarray
from scipy.sparse import csc_array, csr_array

type Slice = ndarray | csc_array | csr_array
