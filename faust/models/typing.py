import abc
import random
import string
import sys
from datetime import datetime
from decimal import Decimal
from enum import Enum
from itertools import count
from types import FrameType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from mode.utils.objects import (
    DICT_TYPES,
    LIST_TYPES,
    SET_TYPES,
    TUPLE_TYPES,
    _remove_optional,
    cached_property,
    is_optional,
    is_union,
    qualname,
)
from mode.utils.typing import Counter
from typing_extensions import Final

from faust.types.models import (
    CoercionHandler,
    CoercionMapping,
    IsInstanceArgT,
    ModelT,
)
from faust.utils.iso8601 import parse as parse_iso8601
from faust.utils.json import str_to_decimal

__all__ = ['TypeExpression']

T = TypeVar('T')
MISSING: Final = object()
TUPLE_NAME_COUNTER = count(0)
JSON_TYPES: IsInstanceArgT = (  # XXX FIXME
    str,
    list,
    dict,
    int,
    float,
    Decimal,
)

_getframe: Callable[[int], FrameType] = getattr(sys, '_getframe')  # noqa


def qualname_to_identifier(s: str) -> str:
    return s.replace(
        '.', '__').replace(
            '@', '__').replace(
                '>', '').replace(
                    '<', '')


class NodeType(Enum):
    ROOT = 'ROOT'
    UNION = 'UNION'
    ANY = 'ANY'
    LITERAL = 'LITERAL'
    DATETIME = 'DATETIME'
    DECIMAL = 'DECIMAL'
    NAMEDTUPLE = 'NAMEDTUPLE'
    TUPLE = 'TUPLE'
    SET = 'SET'
    DICT = 'DICT'
    LIST = 'LIST'
    MODEL = 'MODEL'
    USER = 'USER'


USER_TYPES = frozenset({
    NodeType.DATETIME,
    NodeType.DECIMAL,
    NodeType.USER,
    NodeType.MODEL,
})

GENERIC_TYPES = frozenset({
    NodeType.TUPLE,
    NodeType.SET,
    NodeType.DICT,
    NodeType.LIST,
    NodeType.NAMEDTUPLE,
})

NONFIELD_TYPES = frozenset({
    NodeType.NAMEDTUPLE,
    NodeType.MODEL,
    NodeType.USER,
})


class TypeInfo(NamedTuple):
    type: Type
    poly_type: Type
    args: Tuple
    is_optional: bool


class Variable:

    def __init__(self, name: str, *,
                 getitem: Any = None) -> None:
        self.name = name
        self.getitem = getitem

    def __str__(self) -> str:
        if self.getitem is not None:
            return f'{self.name}[{self.getitem}]'
        else:
            return self.name

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self}>'

    def __getitem__(self, name: Any) -> 'Variable':
        return self.clone(getitem=name)

    def clone(self, *,
              name: str = None,
              getitem: Any = MISSING) -> 'Variable':
        return type(self)(
            name=name if name is not None else self.name,
            getitem=getitem if getitem is not MISSING else self.getitem,
        )

    def next_identifier(self) -> 'Variable':
        name = self.name
        next_ord = ord(name[-1]) + 1
        if next_ord > 122:
            name = name + 'a'
        return self.clone(
            name=name[:-1] + chr(next_ord),
            getitem=None,
        )


class Node(abc.ABC):
    BUILTIN_TYPES: ClassVar[Dict[NodeType, Type['Node']]] = {}
    type: ClassVar[NodeType]

    compatible_types: IsInstanceArgT

    expr: Type
    root: 'RootNode'

    def __init_subclass__(self) -> None:
        self._register()

    @classmethod
    def _register(cls) -> None:
        # NOTE The order in which we define node classes
        # matter.
        # For example LiteralNode must always be declared first
        # as issubclass(str, Sequence) is True and ListNode will eat
        # it up.  The same is true for tuples: NamedTupleNode must
        # be defined before TupleNode.
        cls.BUILTIN_TYPES[cls.type] = cls

    @classmethod
    def _is_compatible(cls, info: TypeInfo) -> Tuple[bool, Optional[Type]]:
        if cls.compatible_types:
            try:
                if cls._issubclass(info.poly_type, cls.compatible_types):
                    return True, info.type
            except TypeError:
                pass
        return False, None

    @classmethod
    def _issubclass(cls, typ: Type, types: IsInstanceArgT) -> bool:
        try:
            return issubclass(typ, types)
        except TypeError:
            return False

    @classmethod
    def inspect_type(cls, typ: Type) -> TypeInfo:
        optional = is_optional(typ)
        args, poly_type = _remove_optional(typ, find_origin=True)
        return TypeInfo(typ, poly_type, tuple(args), optional)

    def __init__(self, expr: Type, root: 'RootNode' = None) -> None:
        assert root is not None
        assert root.type is NodeType.ROOT
        self.expr: Type = expr
        self.root = root
        self.root.type_stats[self.type] += 1
        assert self.root.type_stats[NodeType.ROOT] == 1
        self.__post_init__()

    def __post_init__(self) -> None:
        ...

    def random_identifier(self, n: int = 8) -> str:
        return ''.join(random.choice(string.ascii_letters) for _ in range(n))

    @abc.abstractmethod
    def compile(self, var: Variable, *args: Type) -> str:
        ...


class AnyNode(Node):
    type = NodeType.ANY
    compatible_types = ()

    @classmethod
    def _is_compatible(cls, info: TypeInfo) -> Tuple[bool, Optional[Type]]:
        if info.type is Any:
            return True, info.type
        return False, None

    def compile(self, var: Variable, *args: Type) -> str:
        return f'{var}'


class UnionNode(Node):
    type = NodeType.UNION
    compatible_types = ()

    @classmethod
    def _is_compatible(cls, info: TypeInfo) -> Tuple[bool, Optional[Type]]:
        return False, None


class LiteralNode(Node):
    type = NodeType.LITERAL
    compatible_types = (str, bytes, float, int)

    def compile(self, var: Variable, *args: Type) -> str:
        return f'{var}'


class DecimalNode(Node):
    type = NodeType.DECIMAL
    compatible_types = (Decimal,)

    def compile(self, var: Variable, *args: Type) -> str:
        self.root.extra_locals.setdefault('_Decimal_', self._maybe_coerce)
        return f'_Decimal_({var})'

    @staticmethod
    def _maybe_coerce(value: Union[str, Decimal] = None) -> Optional[Decimal]:
        if value is not None:
            if not isinstance(value, Decimal):
                return str_to_decimal(value)
            return value
        return None


class DateTimeNode(Node):
    type = NodeType.DATETIME
    compatible_types = (datetime,)

    def compile(self, var: Variable, *args: Type) -> str:
        self.root.extra_locals.setdefault(
            '_iso8601_parse_', self._maybe_coerce)
        return f'_iso8601_parse_({var})'

    @staticmethod
    def _maybe_coerce(
            value: Union[str, datetime] = None) -> Optional[datetime]:
        if value is not None:
            if isinstance(value, str):
                return parse_iso8601(value)
            return value
        return None


class NamedTupleNode(Node):
    type = NodeType.NAMEDTUPLE
    compatible_types = TUPLE_TYPES

    @classmethod
    def _is_compatible(cls, info: TypeInfo) -> Tuple[bool, Optional[Type]]:
        typ = info.type
        try:
            is_tuple = issubclass(info.poly_type, cls.compatible_types)
        except TypeError:
            pass
        else:
            if (is_tuple and
                    '_asdict' in typ.__dict__ and
                    '_make' in typ.__dict__ and
                    '_fields_defaults' in typ.__dict__):
                return True, info.type
        return False, None

    def compile(self, var: Variable, *args: Type) -> str:
        self.root.extra_locals.setdefault(self.local_name, self.expr)
        tup = self.expr
        fields = ', '.join(
            '{0}={1}'.format(
                field, self.root.compile(var[i], typ))
            for i, (field, typ) in enumerate(tup._field_types.items())
        )
        return f'{self.local_name}({fields})'

    def next_namedtuple_name(self, typ: Type[Tuple]) -> str:
        num = next(TUPLE_NAME_COUNTER)
        return f'namedtuple_{num}_{typ.__name__}'

    @cached_property
    def local_name(self) -> str:
        return self.next_namedtuple_name(self.expr)


class TupleNode(Node):
    type = NodeType.TUPLE
    compatible_types = TUPLE_TYPES

    def compile(self, var: Variable, *args: Type) -> str:
        if not args:
            return self._compile_untyped_tuple(var)
        for position, arg in enumerate(args):
            if arg is Ellipsis:
                assert position == 1
                return self._compile_vararg_tuple(var, args[0])
        return self._compile_tuple_literal(var, *args)

    def _compile_tuple_literal(self, var: Variable, *member_args: Type) -> str:
        source = '(' + ', '.join(
            self.root.compile(var[i], arg)
            for i, arg in enumerate(member_args)) + ')'
        if ',' not in source:
            return source[:-1] + ',)'
        return source

    def _compile_untyped_tuple(self, var: Variable) -> str:
        return f'tuple({var})'

    def _compile_vararg_tuple(self, var: Variable, member_type: Type) -> str:
        item_var = var.next_identifier()
        handler = self.root.compile(item_var, member_type)
        return f'tuple({handler} for {item_var} in {var})'


class SetNode(Node):
    type = NodeType.SET
    compatible_types = SET_TYPES

    def compile(self, var: Variable, *args: Type) -> str:
        if not args:
            return f'set({var})'
        return self._build_set_expression(var, *args)

    def _build_set_expression(self, var: Variable, member_type: Type) -> str:
        member_var = var.next_identifier()
        handler = self.root.compile(member_var, member_type)
        return f'{{{handler} for {member_var} in {var}}}'


class DictNode(Node):
    type = NodeType.DICT
    compatible_types = DICT_TYPES

    def compile(self, var: Variable, *args: Type) -> str:
        if not args:
            return f'dict({var})'
        return self._build_dict_expression(var, *args)

    def _build_dict_expression(self, var: Variable,
                               key_type: Type, value_type: Type) -> str:
        key_var = var.next_identifier()
        value_var = key_var.next_identifier()
        key_handler = self.root.compile(key_var, key_type)
        value_handler = self.root.compile(value_var, value_type)
        return (f'{{{key_handler}: {value_handler} '
                f'for {key_var}, {value_var} in {var}.items()}}')


class ListNode(Node):
    type = NodeType.LIST
    compatible_types = LIST_TYPES

    def compile(self, var: Variable, *args: Type) -> str:
        if not args:
            return f'list({var})'
        return self._build_list_expression(var, *args)

    def _build_list_expression(self, var: Variable, item_type: Type) -> str:
        item_var = var.next_identifier()
        handler = self.root.compile(item_var, item_type)
        return f'[{handler} for {item_var} in {var}]'


class ModelNode(Node):
    type = NodeType.MODEL
    compatible_types = ()

    @classmethod
    def _is_compatible(cls, info: TypeInfo) -> Tuple[bool, Optional[Type]]:
        args = getattr(info.type, '__args__', ())
        if is_union(info.type) and len(args):
            for arg in args:
                arginfo = cls.inspect_type(arg)
                if cls._is_model(arginfo.type):
                    return True, arginfo.type
        if cls._is_model(info.type):
            return True, info.type
        return False, None

    @classmethod
    def _is_model(cls, typ: Type) -> bool:
        try:
            if issubclass(typ, ModelT):
                return True
        except TypeError:
            pass
        return False

    def compile(self, var: Variable, *args: Type) -> str:
        from .base import Model
        try:
            namespace = self.expr._options.namespace
        except AttributeError:
            # abstract model
            model_name = '_Model_'
            self.root.extra_locals.setdefault(model_name, Model)
        else:
            model_name = qualname_to_identifier(namespace)
            self.root.extra_locals.setdefault(model_name, self.expr)
        return f'{model_name}._from_data_field({var})'


class UserNode(Node):
    type = NodeType.USER
    compatible_types = ()
    handler_name: str

    def __init__(self, expr: Type, root: 'RootNode' = None, *,
                 user_types: CoercionMapping = None,
                 handler: CoercionHandler) -> None:
        super().__init__(expr, root)
        self.handler: CoercionHandler = handler
        self.handler_name = qualname_to_identifier(qualname(self.handler))

    def _maybe_coerce(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, JSON_TYPES):
            return self.handler(value)
        return value

    def compile(self, var: Variable, *args: Type) -> str:
        self.root.extra_locals.setdefault(
            self.handler_name, self._maybe_coerce)
        return f'{self.handler_name}({var})'


class RootNode(Node):
    DEFAULT_NODE: ClassVar[Optional[Type['Node']]] = None

    type = NodeType.ROOT
    type_stats: Counter[NodeType]
    user_types: CoercionMapping
    extra_locals: Dict[str, Any]

    @classmethod
    def _register(cls) -> None:
        ...  # we do not register root nodes.

    def __init__(self, expr: Type, root: 'RootNode' = None, *,
                 user_types: CoercionMapping = None) -> None:
        assert self.type == NodeType.ROOT
        self.type_stats = Counter()
        self.user_types = user_types or {}
        self.extra_locals = {}
        super().__init__(expr, root=self)

    def find_compatible_node_or_default(self, info: TypeInfo) -> 'Node':
        node = self.find_compatible_node(info)
        if node is None:
            return self.new_default_node(info.type)
        else:
            return node

    def find_compatible_node(self, info: TypeInfo) -> Optional['Node']:
        for types, handler in self.user_types.items():
            if self._issubclass(info.poly_type, types):
                return UserNode(info.type, root=self.root, handler=handler)
        for node_cls in self.BUILTIN_TYPES.values():
            is_compatible, expr = node_cls._is_compatible(info)
            if is_compatible and expr is not None:
                return node_cls(expr, root=self.root)
        return None

    def new_default_node(self, typ: Type) -> 'Node':
        if self.DEFAULT_NODE is None:
            raise NotImplementedError(
                f'Node of type {type(self).__name__} has no default node type')
        return self.DEFAULT_NODE(typ, root=self.root)


class TypeExpression(RootNode):
    DEFAULT_NODE = LiteralNode

    type = NodeType.ROOT
    compatible_types = ()

    def as_function(self,
                    *,
                    name: str = 'expr',
                    argument_name: str = 'a',
                    stacklevel: int = 1,
                    locals: Dict[str, Any] = None,
                    globals: Dict[str, Any] = None) -> Callable[[T], T]:
        sourcecode = self.as_string(name=name, argument_name=argument_name)
        if locals is None or globals is None and stacklevel:
            frame = _getframe(stacklevel)
            globals = frame.f_globals if globals is None else globals
            locals = frame.f_locals if locals is None else locals
        new_globals = dict(globals or {})
        new_globals.update(self.extra_locals)
        return self._build_function(
            name, sourcecode,
            locals={} if locals is None else locals,
            globals=new_globals,
        )

    def as_string(self, *,
                  name: str = 'expr',
                  argument_name: str = 'a') -> str:
        expression = self.as_comprehension(argument_name)
        return self._build_function_source(
            name,
            args=[argument_name],
            body=[f'return {expression}'],
        )

    def as_comprehension(self, argument_name: str = 'a') -> str:
        return self.compile(Variable(argument_name), self.expr)

    def compile(self, var: Variable, *args: Type) -> str:
        return self._build_expression(var, *args)

    def _build_expression(self, var: Variable, typ: Type) -> str:
        type_info = self.inspect_type(typ)
        node = self.find_compatible_node_or_default(type_info)
        res = node.compile(var, *type_info.args)
        if type_info.is_optional:
            return f'({res} if {var} is not None else None)'
        else:
            return res

    def _build_function(self, name: str, source: str,
                        *,
                        return_type: Any = MISSING,
                        globals: Dict[str, Any] = None,
                        locals: Dict[str, Any] = None) -> Callable:
        """Generate a function from Python."""
        assert locals is not None
        if return_type is not MISSING:
            locals['_return_type'] = return_type
        exec(source, globals, locals)
        obj = locals[name]
        obj.__sourcecode__ = source
        return cast(Callable, obj)

    def _build_function_source(self,
                               name: str,
                               args: List[str],
                               body: List[str],
                               *,
                               return_type: Any = MISSING,
                               argsep: str = ', ') -> str:
        return_annotation = ''
        if return_type is not MISSING:
            return_annotation = '->_return_type'
        bodys = '\n'.join(f'  {b}' for b in body)
        return f'def {name}({argsep.join(args)}){return_annotation}:\n{bodys}'

    @property
    def has_models(self) -> bool:
        return bool(self.type_stats[NodeType.MODEL])

    @property
    def has_custom_types(self) -> bool:
        return bool(self.type_stats.keys() & USER_TYPES)

    @property
    def has_generic_types(self) -> bool:
        return bool(self.type_stats.keys() & GENERIC_TYPES)

    @property
    def has_nonfield_types(self) -> bool:
        return bool(self.type_stats.keys() & NONFIELD_TYPES)
