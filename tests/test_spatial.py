from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

import numpy as np
import pandas as pd
from topact import spatial

MAX_VALUE = np.iinfo('int64').max

@st.composite
def dimensions_and_genes_and_counts(draw):
    xmin = draw(st.integers())
    ymin = draw(st.integers())
    xlength = draw(st.integers(min_value=1, max_value=10))
    ylength = draw(st.integers(min_value=1, max_value=10))
    xmax = xmin + xlength - 1
    ymax = ymin + ylength - 1

    genes = draw(st.lists(st.text(min_size=1), unique=True, max_size=10, min_size=1))
    num_genes = len(genes)

    counts = draw(arrays(int,
                         (xlength,ylength,num_genes),
                         elements=st.integers(min_value=0, max_value=MAX_VALUE),
                         fill=st.just(0))
                         )

    return xmin, xmax, ymin, ymax, genes, counts


@given(st.lists(st.integers()))
def test_split_undoes_combine(coords):
    assert tuple(coords) == spatial.split_coords(spatial.combine_coords(coords))


@given(st.tuples(st.integers()), st.tuples(st.integers()))
def combine_is_injective(coords1, coords2):
    combined1 = spatial.combine_coords(coords1)
    combined2 = spatial.combine_coords(coords2)
    assert (coords1 == coords2) == (combined1 == combined2)


@given(st.lists(st.integers(), min_size=1))
def test_first_coord(coords):
    assert spatial.first_coord(spatial.combine_coords(coords)) == coords[0]


@given(st.lists(st.integers(), min_size=2))
def test_second_coord(coords):
    assert spatial.second_coord(spatial.combine_coords(coords)) == coords[1]


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_cartesian_product_shape(x, y):
    cp = spatial.cartesian_product(x, y)
    assert cp.shape == (len(x) * len(y), 2)


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_cartesian_product_entries_are_in_the_product(x, y):
    cp = spatial.cartesian_product(x, y)
    for a, b in cp:
        assert a in x and b in y


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_cartesian_product_gets_all_entries(x, y):
    cp = spatial.cartesian_product(x, y)
    for a in x:
        for b in y:
            assert (a, b) in cp


@given(st.integers(), st.integers(), st.integers(min_value=0, max_value=500), st.integers(), st.integers(min_value=0, max_value=100), st.integers(), st.integers(min_value=0, max_value=100))
def test_square_nbhd_in_grid(x, y, scale, x_min, x_length, y_min, y_length):
    x_max, y_max = x_min + x_length - 1, y_min + y_length - 1
    nbhd = spatial.square_nbhd((x,y), scale, (x_min, x_max), (y_min, y_max))

    for i, j in nbhd:
        assert x_min <= i <= x_max
        assert y_min <= j <= y_max


@given(st.integers(), st.integers(), st.integers(min_value=0, max_value=500), st.integers(), st.integers(min_value=0, max_value=100), st.integers(), st.integers(min_value=0, max_value=100))
def test_square_nbhd_in_range(x, y, scale, x_min, x_length, y_min, y_length):
    x_max, y_max = x_min + x_length - 1, y_min + y_length - 1
    nbhd = spatial.square_nbhd((x,y), scale, (x_min, x_max), (y_min, y_max))

    for i, j in nbhd:
        assert abs(i - x) <= scale
        assert abs(j - y) <= scale

@given(st.integers(), st.integers(), st.integers(min_value=0, max_value=500), st.integers(), st.integers(min_value=0, max_value=100), st.integers(), st.integers(min_value=0, max_value=100))
def test_square_nbhd_has_everything(x, y, scale, x_min, x_length, y_min, y_length):
    x_max, y_max = x_min + x_length - 1, y_min + y_length - 1
    nbhd = spatial.square_nbhd((x,y), scale, (x_min, x_max), (y_min, y_max))

    nbhd = set(nbhd)

    for i in range(x - scale, x + scale+1):
        for j in range(y-scale, y + scale+1):
            if x_min <= i <= x_max and y_min <= j <= y_max:
                assert (i,j) in nbhd

@given(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000), st.integers(min_value=0, max_value=500), st.integers(min_value=-1000, max_value=1000), st.integers(min_value=0, max_value=100), st.integers(min_value=-1000, max_value=1000), st.integers(min_value=0, max_value=100))
def test_square_nbhd_vec_in_grid(x, y, scale, x_min, x_length, y_min, y_length):
    x_max, y_max = x_min + x_length - 1, y_min + y_length - 1
    nbhd = spatial.square_nbhd_vec((x,y), scale, (x_min, x_max), (y_min, y_max))

    for i, j in nbhd:
        assert x_min <= i <= x_max
        assert y_min <= j <= y_max


@given(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000), st.integers(min_value=0, max_value=500), st.integers(min_value=-1000, max_value=1000), st.integers(min_value=0, max_value=100), st.integers(min_value=-1000, max_value=1000), st.integers(min_value=0, max_value=100))
def test_square_nbhd_vec_in_range(x, y, scale, x_min, x_length, y_min, y_length):
    x_max, y_max = x_min + x_length - 1, y_min + y_length - 1
    nbhd = spatial.square_nbhd_vec((x,y), scale, (x_min, x_max), (y_min, y_max))

    for i, j in nbhd:
        assert abs(i - x) <= scale
        assert abs(j - y) <= scale

@given(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000), st.integers(min_value=0, max_value=500), st.integers(min_value=-1000, max_value=1000), st.integers(min_value=0, max_value=100), st.integers(min_value=-1000, max_value=1000), st.integers(min_value=0, max_value=100))
def test_square_nbhd_vec_has_everything(x, y, scale, x_min, x_length, y_min, y_length):
    x_max, y_max = x_min + x_length - 1, y_min + y_length - 1
    nbhd = spatial.square_nbhd_vec((x,y), scale, (x_min, x_max), (y_min, y_max))

    nbhd = set(list(map(tuple, list(nbhd))))

    for i in range(x - scale, x + scale+1):
        for j in range(y-scale, y + scale+1):
            if x_min <= i <= x_max and y_min <= j <= y_max:
                assert (i,j) in nbhd

@given(dimensions_and_genes_and_counts())
def test_expressiongrid_matrix(dagac):
    xmin, xmax, ymin, ymax, genes, counts = dagac
    xlength = xmax - xmin + 1
    ylength = ymax - ymin + 1
    xs = sum([[x] * len(genes) * ylength for x in range(xmin, xmax+1)], [])
    ys = sum([[y] * len(genes) for y in range(ymin, ymax+1)], []) * xlength
    data = {'x': xs,
            'y': ys,
            'gene': genes * (xlength * ylength),
            'count': list(counts.flatten())}


    table = pd.DataFrame(data)

    expression_grid = spatial.ExpressionGrid(table, genes)

    assert expression_grid.matrix.shape == (xlength*ylength, len(genes))

    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            for g in range(len(genes)):
                assert np.array(expression_grid.expression((x,y)))[0][g] == counts[x-xmin, y-ymin, g]
