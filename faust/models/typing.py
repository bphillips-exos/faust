import abc
import sys
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
)
from mode.utils.typing import Counter

from faust.types.models import ModelT

__all__ = ['TypeExpression']

T = TypeVar('T')
MISSING = object()
TUPLE_NAME_COUNTER = count(0)

_getframe: Callable[[int], FrameType] = getattr(sys, '_getframe')



class NodeType(Enum):
    ROOT = 'ROOT'
    UNION = 'UNION'
    ANY = 'ANY'
    LITERAL = 'LITERAL'
    NAMEDTUPLE = 'NAMEDTUPLE'
    TUPLE = 'TUPLE'
    SET = 'SET'
    DICT = 'DICT'
    LIST = 'LIST'
    MODEL = 'MODEL'
    USER = 'USER'


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
    TYPES: ClassVar[Dict[NodeType, Type['Node']]] = {}
    DEFAULT_NODE: ClassVar[Optional[Type['Node']]] = None
    compatible_types: ClassVar[Tuple[Type, ...]]
    type: ClassVar[NodeType]

    type_stats: Counter[NodeType]

    expr: Type
    root: 'Node'

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
        cls.TYPES[cls.type] = cls

    @classmethod
    def _is_compatible(cls, info: TypeInfo) -> Tuple[bool, Optional[Type]]:
        if cls.compatible_types:
            try:
                if issubclass(info.poly_type, cls.compatible_types):
                    return True, info.type
            except TypeError:
                pass
        return False, None

    def __init__(self, expr: Type, root: 'Node' = None) -> None:
        self.expr: Type = expr
        if root is None:
            assert self.type == NodeType.ROOT
            self.root = self
            self.type_stats = Counter()
        else:
            self.root = root
        self.root.type_stats[self.type] += 1
        self.__post_init__()

    def __post_init__(self) -> None:
        ...

    @classmethod
    def inspect_type(cls, typ: Type) -> TypeInfo:
        optional = is_optional(typ)
        args, poly_type = _remove_optional(typ, find_origin=True)
        return TypeInfo(typ, poly_type, tuple(args), optional)

    def find_compatible_node_or_default(self, info: TypeInfo) -> 'Node':
        node = self.find_compatible_node(info)
        if node is None:
            return self.new_default_node(info.type)
        else:
            return node

    def find_compatible_node(self, info: TypeInfo) -> Optional['Node']:
        for node_cls in self.TYPES.values():
            is_compatible, expr = node_cls._is_compatible(info)
            if is_compatible:
                return node_cls(expr, root=self.root)
        return None

    def new_default_node(self, typ: Type) -> 'Node':
        if self.DEFAULT_NODE is None:
            raise NotImplementedError(
                f'Node of type {type(self).__name__} has no default node type')
        return self.DEFAULT_NODE(typ, root=self.root)

    @abc.abstractmethod
    def compile(self, var: Variable, *args: Type) -> str:
        ...

    @property
    def extra_locals(self) -> Dict[str, Any]:
        raise NotImplementedError('Only on root nodes.')


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
        return (
            '(' + ', '.join(
                self.root.compile(var[i], arg)
                for i, arg in enumerate(args)) + ')'
        )


class SetNode(Node):
    type = NodeType.SET
    compatible_types = SET_TYPES

    def compile(self, var: Variable, *args: Type) -> str:
        return self._build_set_expression(var, *args)

    def _build_set_expression(self, var: Variable, member_type: Type) -> str:
        member_var = var.next_identifier()
        handler = self.root.compile(member_var, member_type)
        return f'{{{handler} for {member_var} in {var}}}'


class DictNode(Node):
    type = NodeType.DICT
    compatible_types = DICT_TYPES

    def compile(self, var: Variable, *args: Type) -> str:
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
        if is_union(info.type) and len(args) > 2:
            for arg in args:
                arginfo = cls.inspect_type(arg)
                if cls._is_model(arginfo.type):
                    print("IS COMPATIBLE")
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
            return False

    def compile(self, var: Variable, *args: Type) -> str:
        from .base import Model, registry
        try:
            namespace = self.expr._options.namespace
        except AttributeError:
            # abstract model
            model_name = '_Model_'
            self.root.extra_locals.setdefault(model_name, Model)
        else:
            model_name = namespace.replace(
                '.', '__').replace(
                    '@', '__').replace(
                        '>', '').replace(
                            '<', '')
            self.root.extra_locals.setdefault(model_name, self.expr)
        return f'({model_name}.from_data({var}) if {var} is not None else None)'


class UserNode(Node):
    type = NodeType.USER
    compatible_types = ()

    def compile(self, var: Variable, *args: Type) -> str:
        return f'{var}'


class TypeExpression(Node):

    DEFAULT_NODE = UserNode
    type = NodeType.ROOT
    compatible_types = ()
    _extra_locals: Dict[str, Any]

    @classmethod
    def _register(cls) -> None:
        ...

    def __post_init__(self) -> None:
        self._extra_locals = {}

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
        new_globals = dict(globals)
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
    def extra_locals(self) -> Dict[str, Any]:
        return self._extra_locals


def comprehension_from_type_expression(
        typ: Type[T], *,
        locals: Dict[str, Any] = None,
        globals: Dict[str, Any] = None) -> Callable[[T], T]:
    return TypeExpression(typ).as_function(
        locals=locals,
        globals=globals,
    )
