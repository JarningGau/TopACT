import itertools

import numpy as np
from numpy.testing import assert_array_equal
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from topact import densetools

matrix_strategy = arrays(int,
                         array_shapes(min_dims=2, max_dims=2),
                         fill=st.just(0)
                         )


@given(arrays(dtype=int, shape=array_shapes(min_dims=1, max_dims=1), fill=st.just(0)))
def test_first_nonzero_1d_is_nonzero(vector):
    index = densetools.first_nonzero_1d(vector)
    if index != -1:
        assert vector[index] != 0


@given(array_shapes(min_dims=1, max_dims=1))
def test_first_nonzero_1d_on_zero(length):
    zeros = np.zeros(length)
    assert densetools.first_nonzero_1d(zeros) == -1


@given(arrays(dtype=int, shape=array_shapes(min_dims=1, max_dims=1), fill=st.just(0)))
def test_first_nonzero_1d_is_first(vector):
    index = densetools.first_nonzero_1d(vector)
    assert not any(vector[:None if index == -1 else index])


@given(matrix_strategy)
def test_first_nonzero_2d_is_right_shape_rows(array):
    assert len(densetools.first_nonzero_2d(array, 1)) == array.shape[0]


@given(matrix_strategy)
def test_first_nonzero_2d_is_right_shape_cols(array):
    assert len(densetools.first_nonzero_2d(array, 0)) == array.shape[1]


@given(matrix_strategy)
def test_last_nonzero_2d_is_right_shape_rows(array):
    assert len(densetools.last_nonzero_2d(array, 1)) == array.shape[0]


@given(matrix_strategy)
def test_last_nonzero_2d_is_right_shape_cols(array):
    assert len(densetools.last_nonzero_2d(array, 0)) == array.shape[1]


@given(matrix_strategy)
def test_first_nonzero_2d_entries_are_nonzero_rows(array):
    firsts = densetools.first_nonzero_2d(array, axis=1)
    for i, j in enumerate(firsts):
        if j != -1:
            assert array[i, j] != 0


@given(matrix_strategy)
def test_first_nonzero_2d_entries_are_nonzero_cols(array):
    firsts = densetools.first_nonzero_2d(array, axis=0)
    for j, i in enumerate(firsts):
        if i != -1:
            assert array[i, j] != 0


@given(matrix_strategy)
def test_last_nonzero_2d_entries_are_nonzero_rows(array):
    lasts = densetools.last_nonzero_2d(array, axis=1)
    for i, j in enumerate(lasts):
        if j != array.shape[1]:
            assert array[i, j] != 0


@given(matrix_strategy)
def test_last_nonzero_2d_entries_are_nonzero_cols(array):
    lasts = densetools.last_nonzero_2d(array, axis=0)
    for j, i in enumerate(lasts):
        if i != array.shape[0]:
            assert array[i, j] != 0


@given(matrix_strategy)
def test_first_nonzero_2d_entries_are_first_rows(array):
    firsts = densetools.first_nonzero_2d(array, axis=1)
    for i, j in enumerate(firsts):
        assert not any(array[i, :None if j == -1 else j])


@given(matrix_strategy)
def test_first_nonzero_2d_entries_are_first_cols(array):
    firsts = densetools.first_nonzero_2d(array, axis=0)
    for j, i in enumerate(firsts):
        assert not any(array[:None if i == -1 else i, j])


@given(matrix_strategy)
def test_last_nonzero_2d_entries_are_last_rows(array):
    lasts = densetools.last_nonzero_2d(array, axis=1)
    for i, j in enumerate(lasts):
        assert not any(array[i, j+1:])


@given(matrix_strategy)
def test_last_nonzero_2d_entries_are_last_cols(array):
    lasts = densetools.last_nonzero_2d(array, axis=0)
    for j, i in enumerate(lasts):
        assert not any(array[i+1:, j])


@given(st.integers())
def test_get_pad(shift):
    pad = densetools._get_pad(shift)
    if shift < 0:
        assert pad == (0, -shift)
    else:
        assert pad == (shift, 0)


@given(st.integers())
def test_get_slice(shift):
    _slice = densetools._get_slice(shift)
    if shift < 0:
        assert _slice == slice(-shift, None)
    elif shift > 0:
        assert _slice == slice(0, -shift)
    else:
        assert _slice == slice(None)


@given(matrix_strategy)
def test_translate_by_zero(matrix):
    assert_array_equal(densetools.translate(matrix, 0, 0), matrix)


@given(matrix_strategy, st.integers(), st.integers())
def test_translate_has_same_shape(matrix, x, y):
    shifted = densetools.translate(matrix, x, y)
    assert shifted.shape == matrix.shape


@given(matrix_strategy, st.integers(), st.integers())
def test_translate_has_correct_entires(matrix, x, y):
    shifted = densetools.translate(matrix, x, y)
    height, width = matrix.shape
    # shifted[i, j] = matrix[i-y, j-x]
    for i, j in itertools.product(range(height), range(width)):
        if 0 <= i - y < height and 0 <= j - x < width:
            assert shifted[i, j] == matrix[i-y, j-x]
        else:
            assert shifted[i, j] == 0


@given(matrix_strategy, st.integers(min_value=0))
def test_pool_has_same_shape(matrix, radius):
    pool = densetools.pool(matrix, radius)
    assert pool.shape == matrix.shape


@given(matrix_strategy, st.integers(min_value=0))
def test_pool_has_correct_values(matrix, radius):
    height, width = matrix.shape
    pool = densetools.pool(matrix, radius)
    for (i, j), pooled in np.ndenumerate(pool):
        i_min = max(i - radius, 0)
        j_min = max(j - radius, 0)
        i_max = min(i + radius + 1, height)
        j_max = min(j + radius + 1, width)
        assert pooled == matrix[i_min:i_max, j_min:j_max].sum()


@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10), st.integers(min_value=-100, max_value=100), st.integers(min_value=1))
def test_density_of_constant(height, width, fill, radius):
    matrix = np.zeros((height, width)) + fill
    np.testing.assert_array_equal(matrix, densetools.density(matrix, radius))
