from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
)
import pytest
from faust.models import Record
from faust.models.typing import (
    TypeExpression,
    comprehension_from_type_expression,
)


class X(Record, namespace='test.X'):
    val: int


class Z(NamedTuple):
    foo: X
    bar: int
    baz: str


class TypeExpressionTest(NamedTuple):
    type_expression: Type
    serialized_data: Any
    expected_result: Any
    expected_comprehension: str


def Xi(i):
    return X(i).dumps()


CASE_TUPLE_LIST_SET_X = TypeExpressionTest(
    type_expression=Tuple[X, List[Set[X]]],
    serialized_data=[
        Xi(0),
        [[Xi(1), Xi(2), Xi(3)],
         [Xi(4), Xi(5), Xi(6)],
         [Xi(7), Xi(8)]]],
    expected_result=(
        X(0),
        [
            {X(1), X(2), X(3)},
            {X(4), X(5), X(6)},
            {X(7), X(8)},
        ],
    ),
    expected_comprehension='''\
(test__X.from_data(a[0]), \
[{test__X.from_data(c) for c in b} \
for b in a[1]])'''
)

CASE_LIST_LIST_X = TypeExpressionTest(
    type_expression=List[List[X]],
    serialized_data=[
        [Xi(1), Xi(2), Xi(3)],
        [Xi(4), Xi(5), Xi(6)],
        [Xi(7), Xi(8), Xi(9)]],
    expected_result=[
        [X(1), X(2), X(3)],
        [X(4), X(5), X(6)],
        [X(7), X(8), X(9)],
    ],
    expected_comprehension='[[test__X.from_data(c) for c in b] for b in a]',
)

CASE_DICT_KEY_STR_VALUE_OPTIONAL_SET_X_COMP = '''
    {b: ({test__X.from_data(d) for d in c} if c is not None else None) \
for b, c in a.items()}
'''.strip()
CASE_DICT_KEY_STR_VALUE_OPTIONAL_SET_X = TypeExpressionTest(
    type_expression=Dict[str, Optional[Set[X]]],
    serialized_data={
        'foo': [Xi(1), Xi(2), Xi(3)],
        'bar': [Xi(3), Xi(4), Xi(5)],
        'baz': [Xi(7), Xi(8)],
        'xaz': None,
        'xuz': [],
    },
    expected_result={
        'foo': {X(1), X(2), X(3)},
        'bar': {X(3), X(4), X(5)},
        'baz': {X(7), X(8)},
        'xaz': None,
        'xuz': set(),
    },
    expected_comprehension=CASE_DICT_KEY_STR_VALUE_OPTIONAL_SET_X_COMP,
)


# List[Dict[Tuple[int, X], List[Mapping[str, Optional[Set[X]]]]]
CASE_COMPLEX1_COMP = '''\
[{(c[0], c[1]): [\
{f: ({test__X.from_data(h) for h in g} if g is not None else None) \
for f, g in e.items()} for e in d] for c, d in b.items()} for b in a]\
'''
CASE_COMPLEX1 = TypeExpressionTest(
    type_expression=List[Dict[
        Tuple[int, Any],
        List[Mapping[str, Optional[Set[X]]]],
    ]],
    serialized_data=[
        {
            (0, 1): [
                {'foo': [Xi(1), Xi(2), Xi(3), Xi(4)]},
                {'bar': [Xi(1), Xi(2)]},
                {'baz': None},
                {'xuz': []},
            ],
            (10, 20): [
                {'moo': [Xi(3), Xi(4), Xi(5), Xi(6)]},
            ],
        },
        {
            (30, 42131): [
                {'xfoo': [Xi(3), Xi(2), Xi(300), Xi(1012012)]},
                {'iasiqwoqwidfaoiwqh': [Xi(3120120), Xi(34894891892398)]},
                {'ieqieiai': None},
                {'uidafjaaoz': []},
            ],
            (3192, 12321): [
                {'moo': [Xi(3), Xi(4), Xi(5), Xi(6)]},
            ],
        },
    ],
    expected_result=[
        {
            (0, 1): [
                {'foo': {X(1), X(2), X(3), X(4)}},
                {'bar': {X(1), X(2)}},
                {'baz': None},
                {'xuz': set()},
            ],
            (10, 20): [
                {'moo': {X(3), X(4), X(5), X(6)}},
            ],
        },
        {
            (30, 42131): [
                {'xfoo': {X(3), X(2), X(300), X(1012012)}},
                {'iasiqwoqwidfaoiwqh': {X(3120120), X(34894891892398)}},
                {'ieqieiai': None},
                {'uidafjaaoz': set()},
            ],
            (3192, 12321): [
                {'moo': {X(3), X(4), X(5), X(6)}},
            ],
        },
    ],
    expected_comprehension=CASE_COMPLEX1_COMP,
)


CASE_DICT_KEY_STR_VALUE_SET_NAMEDTUPLE_Z = TypeExpressionTest(
    type_expression=Dict[str, Set[Z]],
    serialized_data={
        'foo': [[1, 2, 'foo']],
        'bar': [[3, 4, 'bar1'], [5, 6, 'bar2'], [7, 8, 'bar3']],
        'baz': [],
    },
    expected_result={
        'foo': {Z(X(1), 2, 'foo')},
        'bar': {Z(X(3), 4, 'bar1'), Z(X(5), 6, 'bar2'), Z(X(7), 8, 'bar3')},
        'baz': set(),
    },
    expected_comprehension=None,
)


CASE_SCALAR_STR = TypeExpressionTest(
    type_expression=str,
    serialized_data='foo',
    expected_result='foo',
    expected_comprehension='a',
)

CASE_OPTIONAL_SCALAR_STR = TypeExpressionTest(
    type_expression=str,
    serialized_data=None,
    expected_result=None,
    expected_comprehension='a',
)

CASE_SCALAR_INT = TypeExpressionTest(
    type_expression=int,
    serialized_data=100,
    expected_result=100,
    expected_comprehension='a',
)

CASE_LIST_INT = TypeExpressionTest(
    type_expression=List[int],
    serialized_data=[1, 2, 3, 4],
    expected_result=[1, 2, 3, 4],
    expected_comprehension='[b for b in a]',
)

CASE_LIST_OPTIONAL_INT = TypeExpressionTest(
    type_expression=List[Optional[int]],
    serialized_data=[1, 2, 3, None, 4],
    expected_result=[1, 2, 3, None, 4],
    expected_comprehension='[(b if b is not None else None) for b in a]',
)

CASES = [
    CASE_TUPLE_LIST_SET_X,
    CASE_LIST_LIST_X,
    CASE_DICT_KEY_STR_VALUE_OPTIONAL_SET_X,
    CASE_COMPLEX1,
    #CASE_DICT_KEY_STR_VALUE_SET_NAMEDTUPLE_Z,
    #CASE_SCALAR_STR,
    #CASE_OPTIONAL_SCALAR_STR,
    #CASE_SCALAR_INT,
    #CASE_LIST_INT,
    #CASE_LIST_OPTIONAL_INT,
]

# class Y(NamedTuple):
#    x: X
# Dict[str, Set[Y]]


@pytest.mark.parametrize('case', CASES)
def test_compile(case):
    fun = comprehension_from_type_expression(
        case.type_expression,
        globals=globals(),
    )
    assert fun(case.serialized_data) == case.expected_result


@pytest.mark.parametrize('case', CASES)
def test_comprehension(case):
    expr = TypeExpression(case.type_expression)
    if case.expected_comprehension:
        assert expr.as_comprehension() == case.expected_comprehension
