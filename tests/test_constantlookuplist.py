from hypothesis.control import assume
import hypothesis.strategies as st
from hypothesis.stateful import invariant, precondition, rule, RuleBasedStateMachine
import pytest
from topact.constantlookuplist import ConstantLookupList

def option(strategy):
    return st.one_of(st.none(), strategy)

class ListMachine(RuleBasedStateMachine):

    def __init__(self):
        super().__init__()
        self.cll = ConstantLookupList()
        self.lst = []

    @invariant()
    def indices_in_correct_range(self):
        assert all(0 <= self.cll.index(x) <= len(self.cll) for x in self.cll)

    @invariant()
    def elements_have_correct_indices_recorded(self):
        assert all(self.cll[self.cll.index(x)] == x for x in self.cll)

    @invariant()
    def correct_length(self):
        assert len(self.cll) == len(self.lst)

    @invariant()
    def cll_subseq_lst(self):
        assert all(self.cll[i] == x for i, x in enumerate(self.lst))

    @invariant()
    def cll_subset_lst(self):
        assert all(x in self.cll for x in self.lst)

    @invariant()
    def correct_reverse_elements(self):
        assert all(x == y for x, y in zip(reversed(self.lst),
                                          reversed(self.cll),
                                          strict=True))

    @invariant()
    def iter_subsets(self):
        assert all(x in self.cll for x in self.cll)

    @invariant()
    def counts_are_one(self):
        assert all(self.cll.count(x) == 1 for x in self.cll)

    @invariant()
    def correct_truthiness(self):
        assert bool(self.cll) == bool(self.lst)

    @rule(value=st.integers(),
          start=option(st.integers()),
          stop=option(st.integers()))
    def index_same_as_list(self, value, start, stop):
        start = start or 0
        stop = stop or len(self.lst)
        if value in self.lst:
            index = self.lst.index(value)
            if start <= index < stop:
                assert self.cll.index(value, start, stop) == index
            else:
                with pytest.raises(ValueError):
                    _ = self.cll.index(value, start, stop)
        else:
            with pytest.raises(KeyError):
                _ = self.cll.index(value, start, stop)

    @rule(value=st.integers())
    def test_count(self, value):
        if value in self.cll:
            assert self.cll.count(value) == 1
        else:
            assert self.cll.count(value) == 0

    @rule(value=st.integers())
    def append(self, value):
        if value not in self.cll:
            self.cll.append(value)
            self.lst.append(value)

    @precondition(lambda self: self.cll)
    @rule(index=st.integers())
    def delitem(self, index):
        if 0 <= index < len(self.cll):
            del self.cll[index]
            del self.lst[index]

    @precondition(lambda self: self.cll)
    @rule(index=st.integers(), value=st.integers())
    def setvalue(self, index, value):
        assume(-len(self.cll) <= index < len(self.cll))
        if value not in self.cll:
            self.cll[index] = value
            self.lst[index] = value
        elif not self.lst[index] == value:
            with pytest.raises(ValueError):
                self.cll[index] = value

    @rule(index=st.integers(), value=st.integers())
    def insert(self, index, value):
        try:
            if value not in self.cll:
                self.cll.insert(index, value)
            if value in self.cll:
                with pytest.raises(ValueError):
                    self.cll.insert(index, value)
        except OverflowError:
            with pytest.raises(OverflowError):
                self.lst.insert(index, value)
        else:
            if value not in self.lst:
                self.lst.insert(index, value)

    @rule()
    def reverse(self):
        self.lst.reverse()
        self.cll.reverse()


TestConstantLookupLists = ListMachine.TestCase
