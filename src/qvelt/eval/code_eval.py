"""Pure bounded integer-function verifier for original HumanEval-style controls.

This is not arbitrary Python execution or the official HumanEval checker. Only
the allowlisted expression AST is interpreted; generated imports/calls/loops and
side effects are rejected, including inside annotations/defaults/decorators.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from qvelt.utils.manifests import _reject_constant, _unique_mapping


def code_spec(text: str) -> dict[str, Any]:
    import json

    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError("Code target exceeds limit")
    spec = json.loads(
        text, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )
    if not isinstance(spec, dict) or set(spec) != {"entry_point", "arguments", "tests"}:
        raise ValueError("Expected bounded function/test specification")
    arguments = spec["arguments"]
    if (
        not isinstance(spec["entry_point"], str)
        or re.fullmatch(r"[a-z][a-z_]{0,31}", spec["entry_point"]) is None
        or not isinstance(arguments, list)
        or not 1 <= len(arguments) <= 8
        or any(
            not isinstance(a, str) or re.fullmatch(r"[a-z][a-z_]{0,15}", a) is None
            for a in arguments
        )
        or len(set(arguments)) != len(arguments)
        or not isinstance(spec["tests"], list)
        or not 1 <= len(spec["tests"]) <= 64
    ):
        raise ValueError("Invalid function dimensions/names/tests")
    for test in spec["tests"]:
        if (
            not isinstance(test, dict)
            or set(test) != {"args", "expected"}
            or not isinstance(test["args"], list)
            or len(test["args"]) != len(arguments)
            or any(type(v) is not int or abs(v) > 1_000_000 for v in test["args"])
            or type(test["expected"]) not in (int, bool)
            or abs(test["expected"]) > 1_000_000_000
        ):
            raise ValueError("Code tests require bounded integer inputs/scalar golds")
    return spec


def _expression(node: ast.AST, values: dict[str, int]) -> int | bool:
    def bounded(value: int | bool) -> int | bool:
        if abs(value) > 1_000_000_000:
            raise OverflowError("Code arithmetic budget exceeded")
        return value

    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
        return bounded(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, ast.USub | ast.UAdd | ast.Not
    ):
        value = _expression(node.operand, values)
        return (
            not value
            if isinstance(node.op, ast.Not)
            else bounded(-value if isinstance(node.op, ast.USub) else +value)
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, ast.Add | ast.Sub | ast.Mult | ast.FloorDiv | ast.Mod
    ):
        a, b = _expression(node.left, values), _expression(node.right, values)
        value = (
            a + b
            if isinstance(node.op, ast.Add)
            else a - b
            if isinstance(node.op, ast.Sub)
            else a * b
            if isinstance(node.op, ast.Mult)
            else a // b
            if isinstance(node.op, ast.FloorDiv)
            else a % b
        )
        return bounded(value)
    if isinstance(node, ast.IfExp):
        return _expression(
            node.body if _expression(node.test, values) else node.orelse, values
        )
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        a, b = _expression(node.left, values), _expression(node.comparators[0], values)
        op = node.ops[0]
        if isinstance(op, ast.Eq | ast.NotEq | ast.Lt | ast.LtE | ast.Gt | ast.GtE):
            return (
                a == b
                if isinstance(op, ast.Eq)
                else a != b
                if isinstance(op, ast.NotEq)
                else a < b
                if isinstance(op, ast.Lt)
                else a <= b
                if isinstance(op, ast.LtE)
                else a > b
                if isinstance(op, ast.Gt)
                else a >= b
            )
    raise ValueError("Unsupported code expression")


def score_code(text: str, target: str, *, complete: bool) -> dict[str, Any]:
    spec = code_spec(target)
    result: dict[str, Any] = {
        "correct": False,
        "format_valid": False,
        "complete": complete,
        "extracted": None,
        "code_failure": "unsupported_syntax",
    }
    if not complete:
        result["code_failure"] = "truncation"
        return result
    if not isinstance(text, str) or len(text) > 8192:
        return result
    source = (
        text
        if text.lstrip().startswith("def ")
        else f"def {spec['entry_point']}({','.join(spec['arguments'])}):\n" + text
    )
    try:
        tree = ast.parse(source)
        if (
            sum(1 for _ in ast.walk(tree)) > 256
            or len(tree.body) != 1
            or not isinstance(tree.body[0], ast.FunctionDef)
        ):
            return result
        function = tree.body[0]
        args = function.args
        body = function.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if (
            function.name != spec["entry_point"]
            or function.decorator_list
            or function.returns
            or args.posonlyargs
            or args.vararg
            or args.kwarg
            or args.kwonlyargs
            or args.defaults
            or args.kw_defaults
            or [a.arg for a in args.args] != spec["arguments"]
            or any(a.annotation for a in args.args)
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            return result
        # Visit every branch with a grammar whitelist, even if a test never takes it.
        allowed = (
            ast.Expression,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.UnaryOp,
            ast.USub,
            ast.UAdd,
            ast.Not,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Mod,
            ast.IfExp,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        )
        expression = body[0].value
        if any(not isinstance(node, allowed) for node in ast.walk(expression)):
            return result
        result["format_valid"] = True
        checks = []
        for test in spec["tests"]:
            value = _expression(
                expression, dict(zip(spec["arguments"], test["args"], strict=True))
            )
            checks.append(
                type(value) is type(test["expected"]) and value == test["expected"]
            )
        result.update(
            correct=all(checks),
            tests_passed=sum(checks),
            tests=len(checks),
            code_failure=None if all(checks) else "wrong_output",
        )
    except (SyntaxError, ValueError, RecursionError):
        pass
    except (ZeroDivisionError, OverflowError):
        result.update(format_valid=True, code_failure="runtime_error")
    return result
