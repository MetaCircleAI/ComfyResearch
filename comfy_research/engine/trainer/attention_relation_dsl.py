"""Restricted predicate language for attention relation observables.

This module deliberately interprets parsed AST nodes itself.  ``ast.parse`` is
used only to parse the small expression language; user text is never executed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable


class AttentionRelationDslError(ValueError):
    """Validation error that includes the source location when available."""

    def __init__(self, message: str, node: ast.AST | None = None) -> None:
        where = f" at column {node.col_offset + 1}" if node is not None and hasattr(node, "col_offset") else ""
        super().__init__(message + where)


@dataclass(frozen=True)
class AttentionRelationPredicate:
    source: str
    tree: ast.expr
    uses_tokens: bool

    def evaluate(self, *, q: int, k: int, token_ids: list[int] | None, seq_len: int) -> bool:
        return bool(_evaluate(self.tree, q=q, k=k, token_ids=token_ids, seq_len=seq_len))


def compile_attention_relation_predicate(source: object, *, field: str, required: bool) -> AttentionRelationPredicate:
    text = str(source or "").strip()
    if not text:
        if required:
            raise AttentionRelationDslError(f"{field} is required")
        text = "true"
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        column = f" at column {exc.offset}" if exc.offset else ""
        raise AttentionRelationDslError(f"Invalid {field} syntax: {exc.msg}{column}") from exc
    uses_tokens = _validate(tree)
    return AttentionRelationPredicate(source=text, tree=tree, uses_tokens=uses_tokens)


def _validate(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int)) and not isinstance(node.value, float):
            return False
        raise AttentionRelationDslError("Only integer and boolean literals are allowed", node)
    if isinstance(node, ast.Name):
        if node.id in {"true", "false"}:
            return False
        raise AttentionRelationDslError(f"Unknown identifier {node.id!r}", node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        return _validate(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return _validate(node.left) or _validate(node.right)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return any(_validate(value) for value in node.values)
    if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
        return _validate(node.left) or any(_validate(value) for value in node.comparators)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "seq_len":
            if node.args or node.keywords:
                raise AttentionRelationDslError("seq_len() takes no arguments", node)
            return False
        if node.func.id in {"pos", "tok"}:
            if len(node.args) != 1 or node.keywords:
                raise AttentionRelationDslError(f"{node.func.id}() takes exactly one position argument", node)
            arg = node.args[0]
            if node.func.id == "pos" and not (isinstance(arg, ast.Name) and arg.id in {"q", "k"}):
                raise AttentionRelationDslError("pos() accepts only q or k", arg)
            if node.func.id == "tok":
                _validate_position_expr(arg)
                return True
            return False
    raise AttentionRelationDslError("Unsupported expression syntax", node)


def _validate_position_expr(node: ast.AST) -> None:
    if isinstance(node, ast.Name) and node.id in {"q", "k"}:
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _validate_position_expr(node.operand)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        _validate_position_expr(node.left)
        _validate_position_expr(node.right)
        return
    raise AttentionRelationDslError("tok() accepts only a safe integer position expression using q and k", node)


def _negative_position_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value) if node.value < 0 else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if isinstance(value, int) and not isinstance(value, bool):
            return -int(value)
    return None


def _is_pos_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "pos"


def _evaluate(node: ast.AST, *, q: int, k: int, token_ids: list[int] | None, seq_len: int):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return {"q": q, "k": k, "true": True, "false": False}[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, q=q, k=k, token_ids=token_ids, seq_len=seq_len)
        if isinstance(node.op, ast.Not): return not value
        if isinstance(node.op, ast.USub): return -value
        return +value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, q=q, k=k, token_ids=token_ids, seq_len=seq_len)
        right = _evaluate(node.right, q=q, k=k, token_ids=token_ids, seq_len=seq_len)
        return {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right,
                ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right}[type(node.op)]()
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_evaluate(value, q=q, k=k, token_ids=token_ids, seq_len=seq_len) for value in node.values)
        return any(_evaluate(value, q=q, k=k, token_ids=token_ids, seq_len=seq_len) for value in node.values)
    if isinstance(node, ast.Call):
        name = node.func.id  # validated
        if name == "seq_len": return seq_len
        position = _evaluate(node.args[0], q=q, k=k, token_ids=token_ids, seq_len=seq_len)
        if name == "pos": return position
        if token_ids is None:
            raise AttentionRelationDslError("tok() is unsupported for this model input")
        return token_ids[position] if 0 <= position < len(token_ids) else -1
    if isinstance(node, ast.Compare):
        left_node = node.left
        left = _evaluate(left_node, q=q, k=k, token_ids=token_ids, seq_len=seq_len)
        for op, right_node in zip(node.ops, node.comparators):
            right = _evaluate(right_node, q=q, k=k, token_ids=token_ids, seq_len=seq_len)
            # Negative literals are end-relative only in direct pos(...) comparisons.
            if _is_pos_call(left_node) and (negative := _negative_position_literal(right_node)) is not None:
                right = seq_len + negative
            elif _is_pos_call(right_node) and (negative := _negative_position_literal(left_node)) is not None:
                left = seq_len + negative
            ok: Callable[[], bool] = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right, ast.Gt: lambda: left > right,
                ast.GtE: lambda: left >= right}[type(op)]
            if not ok(): return False
            left, left_node = right, right_node
        return True
    raise AssertionError("validated AST unexpectedly reached evaluator")
