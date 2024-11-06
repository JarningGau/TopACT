import scipy.sparse
import numpy as np
from numpy.testing import assert_array_equal
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
import pytest
from topact import sparsetools

OVERFLOW_WARNING = "overflow encountered in multiply"

CONVERTERS = [scipy.sparse.csr_array, scipy.sparse.csc_array]

matrix_strategy = arrays(int,
                         array_shapes(min_dims=2, max_dims=2),
                         fill=st.just(0)
                         )


@st.composite
def matrix_and_factors_strategy(draw, axis):
    matrix = draw(matrix_strategy)
    size = matrix.shape[axis]
    factors = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False),
                            min_size=size, max_size=size
                            )
                   )
    return matrix, factors


@st.composite
def matrix_and_indices_strategy(draw, axis):
    matrix = draw(matrix_strategy)
    size = matrix.shape[axis]
    indices = draw(st.lists(st.integers(min_value=0, max_value=size-1)))
    return matrix, indices


@given(matrix_and_factors_strategy(0))
def test_rescale_rows_has_same_shape(matrix_and_factors):
    """Checks that a rescaled matrix has the same shape"""
    matrix, factors = matrix_and_factors
    for convert in CONVERTERS:
        rescaled = sparsetools.rescale_rows(convert(matrix), factors)
        assert rescaled.shape == matrix.shape


@given(matrix_and_factors_strategy(1))
def test_rescale_columns_has_same_shape(matrix_and_factors):
    """Checks that a rescaled matrix has the same shape"""
    matrix, factors = matrix_and_factors
    for convert in CONVERTERS:
        rescaled = sparsetools.rescale_columns(convert(matrix), factors)
        assert rescaled.shape == matrix.shape


@pytest.mark.filterwarnings(f"ignore:{OVERFLOW_WARNING}")
@given(matrix_and_factors_strategy(0))
def test_rescale_rows_has_correct_values(matrix_and_factors):
    """Checks that a rescaled matrix has correct values"""
    matrix, factors = matrix_and_factors
    for convert in CONVERTERS:
        rescaled = sparsetools.rescale_rows(convert(matrix), factors)
        for i, j in np.ndindex(rescaled.shape):
            assert rescaled[i, j] == matrix[i, j] * factors[i]  # pylint: disable=E1136


@pytest.mark.filterwarnings(f"ignore:{OVERFLOW_WARNING}")
@given(matrix_and_factors_strategy(1))
def test_rescale_columns_has_correct_values(matrix_and_factors):
    """Checks that a rescaled matrix has correct values"""
    matrix, factors = matrix_and_factors
    for convert in CONVERTERS:
        rescaled = sparsetools.rescale_columns(convert(matrix), factors)
        for i, j in np.ndindex(rescaled.shape):
            assert rescaled[i, j] == matrix[i, j] * factors[j]  # pylint: disable=E1136


@given(matrix_strategy)
def test_iterate_sparse_has_right_values(matrix):
    """Checks that all values output are in the matrix at the given place."""
    for convert in CONVERTERS:
        sparse_matrix = convert(matrix)
        for i, j, value in sparsetools.iterate_sparse(sparse_matrix):
            assert sparse_matrix[i, j] == value


@given(matrix_strategy)
def test_iterate_sparse_has_all_values(matrix):
    """Checks that every non-zero value is iterated."""
    nonzero = np.transpose(np.nonzero(matrix))
    for convert in CONVERTERS:
        sparse_matrix = convert(matrix)
        indices = [(i, j) for i, j, _ in sparsetools.iterate_sparse(sparse_matrix)]
        assert all(tuple(x) in indices for x in nonzero)


@given(matrix_and_indices_strategy(1))
def test_filter_cols(matrix_and_keep_indices):
    matrix, keep_indices = matrix_and_keep_indices
    num_cols = matrix.shape[1]
    drop_indices = filter(lambda x: x not in keep_indices, range(num_cols))
    sparse_matrix = scipy.sparse.csr_array(matrix)
    new_matrix = sparsetools.filter_cols(sparse_matrix, list(keep_indices))
    actual = np.delete(matrix, np.array(list(drop_indices), dtype=int), axis=1)
    assert_array_equal(actual,  np.array(new_matrix.todense()))
