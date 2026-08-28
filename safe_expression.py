import ast
import operator
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import torch


_MAX_EXPRESSION_LENGTH = 2048
_MAX_AST_NODES = 128

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}

_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_FUNCTION_NAMES = {
    "abs",
    "absolute",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "ceil",
    "clip",
    "cos",
    "cosh",
    "exp",
    "expm1",
    "floor",
    "log",
    "log10",
    "log1p",
    "log2",
    "max",
    "mean",
    "median",
    "min",
    "maximum",
    "minimum",
    "power",
    "prod",
    "round",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "sum",
    "tan",
    "tanh",
    "where",
}

_NUMPY_FUNCTIONS = {
    name: getattr(np, name)
    for name in _FUNCTION_NAMES
    if hasattr(np, name)
}

_TORCH_FUNCTIONS = {
    name: getattr(torch, name)
    for name in _FUNCTION_NAMES | {"clamp", "lerp", "sigmoid"}
    if hasattr(torch, name)
}

_METHOD_NAMES = {
    "abs",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "exp",
    "expm1",
    "floor",
    "log",
    "log10",
    "log1p",
    "log2",
    "round",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
}

_CONSTANTS = {
    "e": np.e,
    "inf": np.inf,
    "nan": np.nan,
    "pi": np.pi,
}

_MODULE_CONSTANTS = {
    "np": {
        name: getattr(np, name)
        for name in _CONSTANTS
        if hasattr(np, name)
    },
    "torch": {
        name: getattr(torch, name)
        for name in _CONSTANTS
        if hasattr(torch, name)
    },
}


class MathExpression:
    """Evaluate tensor math without exposing Python execution primitives."""

    def __init__(self, expression: str, variable_names: set[str]) -> None:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Math expression must be a non-empty string")
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            raise ValueError("Math expression is too long")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ValueError("Invalid math expression syntax") from error

        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("Math expression is too complex")

        self._expression = tree.body
        self._variable_names = set(variable_names)

    def evaluate(self, variables: Mapping[str, Any]) -> Any:
        unknown_variables = set(variables) - self._variable_names
        if unknown_variables:
            names = ", ".join(sorted(unknown_variables))
            raise ValueError(f"Unexpected math expression variables: {names}")
        return self._evaluate(self._expression, variables)

    def _evaluate(self, node: ast.AST, variables: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            if node.value is Ellipsis:
                return Ellipsis
            if isinstance(node.value, bool):
                return node.value
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed in math expressions")
            if isinstance(node.value, int) and node.value.bit_length() > 256:
                raise ValueError("Integer constant is too large")
            return node.value

        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"Unknown name in math expression: {node.id}")

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in _MODULE_CONSTANTS:
                constant = _MODULE_CONSTANTS[node.value.id].get(node.attr)
                if constant is not None:
                    return constant
            raise ValueError("Attribute access is not allowed in math expressions")

        if isinstance(node, ast.BinOp):
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported binary operator in math expression")
            left = self._evaluate(node.left, variables)
            right = self._evaluate(node.right, variables)
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
                raise ValueError("Power exponent is too large")
            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = _UNARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported unary operator in math expression")
            return operation(self._evaluate(node.operand, variables))

        if isinstance(node, ast.Compare):
            left = self._evaluate(node.left, variables)
            result = None
            for operator_node, comparator in zip(node.ops, node.comparators):
                operation = _COMPARISON_OPERATORS.get(type(operator_node))
                if operation is None:
                    raise ValueError("Unsupported comparison in math expression")
                right = self._evaluate(comparator, variables)
                comparison = operation(left, right)
                result = comparison if result is None else operator.and_(result, comparison)
                left = right
            return result

        if isinstance(node, ast.Call):
            function = self._resolve_function(node.func, variables)
            if any(isinstance(argument, ast.Starred) for argument in node.args):
                raise ValueError("Expanded arguments are not allowed in math expressions")
            if any(keyword.arg is None for keyword in node.keywords):
                raise ValueError("Expanded keyword arguments are not allowed in math expressions")
            arguments = [self._evaluate(argument, variables) for argument in node.args]
            keywords = {
                keyword.arg: self._evaluate(keyword.value, variables)
                for keyword in node.keywords
            }
            return function(*arguments, **keywords)

        if isinstance(node, ast.Subscript):
            value = self._evaluate(node.value, variables)
            index = self._evaluate_slice(node.slice, variables)
            return value[index]

        if isinstance(node, ast.Tuple):
            return tuple(self._evaluate(element, variables) for element in node.elts)

        if isinstance(node, ast.List):
            return [self._evaluate(element, variables) for element in node.elts]

        raise ValueError(f"Unsupported syntax in math expression: {type(node).__name__}")

    def _resolve_function(self, node: ast.AST, variables: Mapping[str, Any]) -> Callable[..., Any]:
        if isinstance(node, ast.Name):
            function = _NUMPY_FUNCTIONS.get(node.id)
            if function is None:
                raise ValueError(f"Function is not allowed in math expression: {node.id}")
            return function

        if not isinstance(node, ast.Attribute):
            raise ValueError("Only named math functions are allowed")

        if isinstance(node.value, ast.Name) and node.value.id in {"np", "torch"}:
            functions = _NUMPY_FUNCTIONS if node.value.id == "np" else _TORCH_FUNCTIONS
            function = functions.get(node.attr)
            if function is None:
                raise ValueError(f"Function is not allowed in math expression: {node.value.id}.{node.attr}")
            return function

        if node.attr not in _METHOD_NAMES:
            raise ValueError(f"Method is not allowed in math expression: {node.attr}")
        value = self._evaluate(node.value, variables)
        function = getattr(value, node.attr, None)
        if function is None or not callable(function):
            raise ValueError(f"Math expression value has no method: {node.attr}")
        return function

    def _evaluate_slice(self, node: ast.AST, variables: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Slice):
            lower = self._evaluate(node.lower, variables) if node.lower is not None else None
            upper = self._evaluate(node.upper, variables) if node.upper is not None else None
            step = self._evaluate(node.step, variables) if node.step is not None else None
            return slice(lower, upper, step)
        return self._evaluate(node, variables)
