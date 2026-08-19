from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


class ConditionError(ValueError):
    pass


def _resolve(value: Any, key: Any) -> Any:
    if isinstance(key, str) and key.startswith("__"):
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    if isinstance(key, str) and hasattr(value, key):
        return getattr(value, key)
    try:
        return value[key]
    except (KeyError, IndexError, TypeError):
        return None


def _evaluate(node: ast.AST, context: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, context)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return context.get(node.id)
    if isinstance(node, ast.Attribute):
        return _resolve(_evaluate(node.value, context), node.attr)
    if isinstance(node, ast.Subscript):
        return _resolve(_evaluate(node.value, context), _evaluate(node.slice, context))
    if isinstance(node, ast.List):
        return [_evaluate(item, context) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item, context) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_evaluate(item, context) for item in node.elts}
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for item in node.values:
                if not _evaluate(item, context):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for item in node.values:
                if _evaluate(item, context):
                    return True
            return False
        raise ConditionError(f"unsupported boolean operator: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        raise ConditionError(f"unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, context)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, context)
            if isinstance(operator, ast.Eq):
                ok = left == right
            elif isinstance(operator, ast.NotEq):
                ok = left != right
            elif isinstance(operator, ast.Lt):
                ok = left < right
            elif isinstance(operator, ast.LtE):
                ok = left <= right
            elif isinstance(operator, ast.Gt):
                ok = left > right
            elif isinstance(operator, ast.GtE):
                ok = left >= right
            elif isinstance(operator, ast.In):
                ok = left in right
            elif isinstance(operator, ast.NotIn):
                ok = left not in right
            elif isinstance(operator, ast.Is):
                ok = left is right
            elif isinstance(operator, ast.IsNot):
                ok = left is not right
            else:
                raise ConditionError(f"unsupported comparison: {type(operator).__name__}")
            if not ok:
                return False
            left = right
        return True
    raise ConditionError(f"unsupported expression node: {type(node).__name__}")


def evaluate_condition(expression: str, context: Mapping[str, Any]) -> bool:
    expression = expression.strip()
    if not expression or expression.lower() == "always":
        return True
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ConditionError(f"invalid condition {expression!r}: {exc.msg}") from exc
    try:
        return bool(_evaluate(tree, context))
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        raise ConditionError(f"condition {expression!r} could not be evaluated: {exc}") from exc
