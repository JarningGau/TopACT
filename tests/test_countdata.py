import hypothesis.strategies as st
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.stateful import (initialize, invariant,
                                 rule, RuleBasedStateMachine)

import numpy as np
import scipy.sparse

from topact.countdata import CountMatrix
from topact import sparsetools

MAX_VALUE = np.iinfo('int64').max

matrix_strategy = arrays(int,
                         array_shapes(min_dims=2, max_dims=2),
                         elements=st.integers(min_value=0,
                                              max_value=MAX_VALUE
                                              ),
                         fill=st.just(0)
                         )


def option(strategy):
    return st.one_of(st.none(), strategy)


@st.composite
def matrix_and_samples_and_genes_strategy(draw):
    matrix = draw(matrix_strategy)
    num_samples, num_genes = matrix.shape
    samples = draw(option(st.lists(st.text(min_size=1),
                          min_size=num_samples,
                          max_size=num_samples,
                          unique=True)))
    genes = draw(option(st.lists(st.text(min_size=1),
                        min_size=num_genes,
                        max_size=num_genes,
                        unique=True)))
    return matrix, samples, genes


class CountMatrixMachine(RuleBasedStateMachine):

    def __init__(self):
        super().__init__()
        self.cm: CountMatrix = None  # pyright: ignore
        self.matrix = self.samples = self.genes = None

    @initialize(msg=matrix_and_samples_and_genes_strategy())
    def init_countmatrix(self, msg):
        matrix, samples, genes = msg
        self.matrix = scipy.sparse.csr_array(matrix)
        self.cm = CountMatrix(self.matrix,
                              samples=samples,
                              genes=genes
                              )

        self.samples = samples or list(map(str, range(matrix.shape[0])))
        self.genes = genes or list(map(str, range(matrix.shape[1])))

    @invariant()
    def num_genes_is_correct(self):
        assert self.cm.num_genes == len(self.genes)

    @invariant()
    def num_samples_is_correct(self):
        assert self.cm.num_samples == len(self.samples)

    @invariant()
    def matrix_is_correct_shape(self):
        assert self.cm.matrix.shape == (self.cm.num_samples, self.cm.num_genes)

    @invariant()
    def expressed_genes_are_expressed_indices(self):
        expressed = self.cm.expressed_genes(output_type='index')
        for gene in expressed:
            assert any(self.matrix[sample, gene] != 0 for  # pyright: ignore
                       sample in range(self.cm.num_samples))

    @invariant()
    def expressed_genes_are_expressed_idents(self):
        expressed = self.cm.expressed_genes(output_type='ident')
        for gene in expressed:
            assert any(self.matrix[sample, self.genes.index(gene)] != 0 for  # pyright: ignore
                       sample in range(self.cm.num_samples))

    @invariant()
    def expressed_genes_are_expressed_per_sample_indices(self):
        for sample, _ in enumerate(self.samples):
            expressed = self.cm.expressed_genes(samples=[sample],
                                                output_type='index')
            for gene in expressed:
                assert self.matrix[sample, gene] != 0  # pyright: ignore

    @invariant()
    def expressed_genes_are_expressed_per_sample_idents(self):
        for sample, _ in enumerate(self.samples):
            expressed = self.cm.expressed_genes(samples=[sample],
                                                output_type='ident')
            for gene in expressed:
                assert self.matrix[sample, self.genes.index(gene)] != 0  # pyright: ignore

    @rule(factors=st.lists(st.floats(allow_nan=False, allow_infinity=False)))
    def rescale_genes(self, factors):
        assume(len(factors) >= self.cm.num_genes)
        factors = factors[:self.cm.num_genes]
        self.cm.rescale_genes(factors)
        sparsetools.rescale_columns(self.matrix, factors)


TestCountMatrix = CountMatrixMachine.TestCase
