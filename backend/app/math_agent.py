"""MathMind mathematical planner and SymPy execution engine.

Responses include LaTeX math delimiters ($...$ / $$...$$). The frontend
must render Markdown math with KaTeX or MathJax for textbook-style display.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import requests
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X, Y, Z = sp.symbols("x y z")
SYMBOLS = {"x": X, "y": Y, "z": Z, "t": sp.Symbol("t")}
MODEL = "qwen2.5:3b"
URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = (2, 18)

TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

OPS = {
    ("equation", "solve"),
    ("calculus", "derivative"), ("calculus", "integral"),
    ("calculus", "limit"), ("calculus", "partial_derivative"),
    ("algebra", "simplify"), ("algebra", "factor"), ("algebra", "expand"),
    ("algebra", "evaluate"),
    ("matrix", "determinant"), ("matrix", "inverse"),
    ("matrix", "transpose"), ("matrix", "rank"),
    ("matrix", "eigenvalues"), ("matrix", "multiply"),
    ("matrix", "add"), ("matrix", "subtract"),
    ("statistics", "mean"), ("statistics", "median"),
    ("statistics", "mode"), ("statistics", "variance"),
    ("statistics", "standard_deviation"),
    ("number_theory", "gcd"), ("number_theory", "lcm"),
    ("number_theory", "prime_factors"),
}


def expr(value: Any) -> sp.Expr:
    """Parse expressions with implicit multiplication and common math notation."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Please provide a mathematical expression.")

    value = value.strip()
    if len(value) > 1000:
        raise ValueError("Expression is too long (maximum 1000 characters).")

    value = (
        value.replace("−", "-")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("π", "pi")
        .replace("²", "**2")
        .replace("³", "**3")
    )

    if re.search(r"[^0-9A-Za-z_+\-*/().,^=\s]", value):
        raise ValueError("Unsupported characters in the expression.")

    local_symbols = dict(SYMBOLS)
    local_symbols.update({
        "pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "log": sp.log, "ln": sp.log, "exp": sp.exp,
        "abs": sp.Abs,
    })

    # Reject unknown words instead of silently treating typos as symbols.
    allowed_names = set(local_symbols)
    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", value))
    unknown = identifiers - allowed_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported symbol(s): {names}. Use x, y, z, or t.")

    try:
        return parse_expr(
            value,
            local_dict=local_symbols,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except (SyntaxError, TypeError, ValueError, sp.SympifyError) as error:
        raise ValueError(f"Could not parse the mathematical expression: {value}") from error


def mat(data: Any) -> sp.Matrix:
    if not isinstance(data, list) or not data or not all(
            isinstance(row, list) and row for row in data
    ):
        raise ValueError("Matrix must be a non-empty list of rows, e.g. [[1,2],[3,4]].")
    if any(len(row) != len(data[0]) for row in data):
        raise ValueError("All matrix rows must have equal lengths.")
    if len(data) > 20 or len(data[0]) > 20:
        raise ValueError("Matrix size is limited to 20×20.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           for row in data for v in row):
        raise ValueError("Matrix entries must be numeric.")
    return sp.Matrix(data)


def matrices(text: str) -> list[list[list[float]]]:
    found = []
    for match in re.findall(r"\[\s*\[[\s\S]*?\]\s*\]", text):
        try:
            data = ast.literal_eval(match)
            mat(data)
            found.append(data)
        except (ValueError, SyntaxError, TypeError):
            pass
    return found[:2]


def _strip_leading(text: str, phrases: list[str]) -> str:
    pattern = r"(?is)^.*?(?:" + "|".join(phrases) + r")\s*"
    return re.sub(pattern, "", text, count=1).strip().rstrip("?.!")


def fallback(text: str) -> dict:
    s = text.strip()
    low = s.lower()
    ms = matrices(s)

    if len(ms) >= 2 and any(k in low for k in ("multiply", "product", "times", " by ")):
        return {"category": "matrix", "operation": "multiply",
                "matrix_a": ms[0], "matrix_b": ms[1]}
    if len(ms) >= 2 and any(k in low for k in ("add", "sum")):
        return {"category": "matrix", "operation": "add",
                "matrix_a": ms[0], "matrix_b": ms[1]}
    if len(ms) >= 2 and any(k in low for k in ("subtract", "difference")):
        return {"category": "matrix", "operation": "subtract",
                "matrix_a": ms[0], "matrix_b": ms[1]}

    matrix_keys = [
        ("determinant", "determinant"), ("det ", "determinant"),
        ("inverse", "inverse"), ("invert", "inverse"),
        ("transpose", "transpose"), ("rank", "rank"),
        ("eigenvalue", "eigenvalues"),
    ]
    if ms:
        for keyword, operation in matrix_keys:
            if keyword in low:
                return {"category": "matrix", "operation": operation, "matrix": ms[0]}

    nums = re.findall(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?![A-Za-z0-9_])", s)
    if len(nums) >= 2 and any(k in low for k in ("gcd", "greatest common divisor")):
        return {"category": "number_theory", "operation": "gcd",
                "values": [int(n) for n in nums]}
    if len(nums) >= 2 and any(k in low for k in ("lcm", "least common multiple")):
        return {"category": "number_theory", "operation": "lcm",
                "values": [int(n) for n in nums]}
    if nums and "prime factor" in low:
        return {"category": "number_theory", "operation": "prime_factors",
                "value": int(nums[-1])}
    if nums and any(k in low for k in ("standard deviation", "std dev")):
        return {"category": "statistics", "operation": "standard_deviation",
                "values": [float(n) for n in nums]}
    if nums and "variance" in low:
        return {"category": "statistics", "operation": "variance",
                "values": [float(n) for n in nums]}
    if nums and "median" in low:
        return {"category": "statistics", "operation": "median",
                "values": [float(n) for n in nums]}
    if nums and ("mean" in low or "average" in low):
        return {"category": "statistics", "operation": "mean",
                "values": [float(n) for n in nums]}
    if nums and "mode" in low:
        return {"category": "statistics", "operation": "mode",
                "values": [float(n) for n in nums]}

    if any(k in low for k in ("partial derivative", "partial differentiation")):
        raw = _strip_leading(s, [
            r"partial derivative of", r"differentiate partially",
            r"partial derivative",
        ])
        var_match = re.search(r"(?i)\b(?:with respect to|w\.r\.t\.?|by)\s+([xyzt])\b", low)
        variable = var_match.group(1) if var_match else "x"
        raw = re.sub(r"(?i)\s*(?:with respect to|w\.r\.t\.?|by)\s+[xyzt]\s*$", "", raw).strip()
        return {"category": "calculus", "operation": "partial_derivative",
                "expression": raw, "variable": variable}

    if any(k in low for k in ("limit", "lim ")):
        match = re.search(
            r"(?i)(?:as\s+)?([xyzt])\s*(?:approaches|approach|->|→)\s*([-+\w.]+)", s
        )
        raw = _strip_leading(s, [r"limit of", r"find the limit of", r"lim"])
        raw = re.sub(
            r"(?i)\s*(?:as\s+)?[xyzt]\s*(?:approaches|approach|->|→)\s*[-+\w.]+",
            "", raw,
        ).strip()
        if match:
            return {"category": "calculus", "operation": "limit",
                    "expression": raw, "variable": match.group(1),
                    "point": match.group(2)}

    if any(k in low for k in ("derivative", "differentiate", "rate of change")):
        raw = _strip_leading(s, [
            r"derivative of", r"differentiate", r"find the derivative of",
            r"rate of change of",
        ])
        var_match = re.search(r"(?i)\b(?:with respect to|w\.r\.t\.?|by)\s+([xyzt])\b", low)
        variable = var_match.group(1) if var_match else "x"
        raw = re.sub(r"(?i)\s*(?:with respect to|w\.r\.t\.?|by)\s+[xyzt]\s*$", "", raw).strip()
        return {"category": "calculus", "operation": "derivative",
                "expression": raw, "variable": variable}

    if any(k in low for k in ("integral", "integrate", "antiderivative")):
        raw = _strip_leading(s, [
            r"integral of", r"integrate", r"find the integral of",
            r"antiderivative of",
        ])
        bounds = re.search(r"(?is)\s+from\s+(.+?)\s+to\s+(.+?)\s*$", raw)
        lower = upper = None
        if bounds:
            raw = raw[:bounds.start()].strip()
            lower, upper = bounds.group(1).strip(), bounds.group(2).strip()
        var_match = re.search(r"(?i)\b(?:with respect to|w\.r\.t\.?|by)\s+([xyzt])\b", low)
        variable = var_match.group(1) if var_match else "x"
        raw = re.sub(r"(?i)\s*(?:with respect to|w\.r\.t\.?|by)\s+[xyzt]\s*$", "", raw).strip()
        result = {"category": "calculus", "operation": "integral",
                  "expression": raw, "variable": variable}
        if bounds:
            result.update({"lower": lower, "upper": upper})
        return result

    if any(k in low for k in ("evaluate", "calculate", "compute")):
        raw = _strip_leading(s, [r"evaluate", r"calculate", r"compute"])
        return {"category": "algebra", "operation": "evaluate", "expression": raw}

    if any(k in low for k in ("factor", "factorise", "factorize")):
        return {"category": "algebra", "operation": "factor",
                "expression": _strip_leading(s, [r"factorise", r"factorize", r"factor"])}
    if any(k in low for k in ("expand", "expansion of")):
        return {"category": "algebra", "operation": "expand",
                "expression": _strip_leading(s, [r"expand", r"expansion of"])}
    if any(k in low for k in ("simplify", "reduce")):
        return {"category": "algebra", "operation": "simplify",
                "expression": _strip_leading(s, [r"simplify", r"reduce"])}

    if any(k in low for k in (
            "solve", "roots of", "root of", "solution of", "solutions of",
            "find the value of", "find x",
    )):
        raw = _strip_leading(s, [
            r"solve", r"roots? of", r"solutions? of",
            r"find the value of", r"find x",
        ])
        raw = re.sub(
            r"(?i)^\s*(?:the\s+)?(?:equation|expression)\s*", "", raw
        ).strip()
        return {"category": "equation", "operation": "solve", "expression": raw}

    return {
        "category": "unknown", "operation": "unknown",
        "clarification": (
            "I can help with equations, algebra, calculus, matrices, statistics, "
            "and number theory. Please provide the expression, matrix, or numbers."
        ),
    }


def plan(text: str, history: list[dict] | None = None) -> dict:
    quick = fallback(text)
    if (quick.get("category"), quick.get("operation")) in OPS:
        return quick

    context = "\n".join(
        f'{m.get("role", "")}: {m.get("content", "")}'
        for m in (history or [])[-4:]
    )[:1200]
    schema = (
        "Return only JSON. Do not solve. Allowed operations: equation/solve(expression); "
        "calculus/derivative(expression,variable), integral(expression,variable), "
        "limit(expression,variable,point), partial_derivative(expression,variable); "
        "algebra/simplify(expression), factor(expression), expand(expression), "
        "evaluate(expression); matrix/determinant,inverse,transpose,rank,eigenvalues(matrix), "
        "multiply/add/subtract(matrix_a,matrix_b); "
        "statistics/mean,median,mode,variance,standard_deviation(values); "
        "number_theory/gcd,lcm(values),prime_factors(value). "
        "For unsupported requests use category unknown, operation unknown, clarification."
    )
    prompt = f"{schema}\nRecent context: {context}\nUser request: {text}"
    try:
        response = requests.post(
            URL,
            json={
                "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
                "options": {"temperature": 0, "num_predict": 180},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        parsed = json.loads(response.json().get("response", "{}"))
        if (parsed.get("category"), parsed.get("operation")) in OPS:
            return parsed
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass
    return quick


def display_math(value: Any) -> str:
    """Compact Unicode representation for the JSON result field."""
    if isinstance(value, sp.Float):
        if value.is_finite and value == int(value):
            return str(int(value))
        return f"{float(value):.10g}"
    if isinstance(value, sp.Basic):
        replacements = {}
        for atom in value.atoms(sp.Float):
            replacements[atom] = (
                sp.Integer(int(atom)) if atom.is_finite and atom == int(atom)
                else sp.Float(f"{float(atom):.10g}")
            )
        value = value.xreplace(replacements) if replacements else value
        text = sp.sstr(value)
        superscripts = str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾")
        text = re.sub(r"\*\*\(([-+]?\d+)\)", lambda m: m.group(1).translate(superscripts), text)
        text = re.sub(r"\*\*([-+]?\d+)", lambda m: m.group(1).translate(superscripts), text)
        text = re.sub(r"sqrt\(([^()]*)\)", r"√(\1)", text)
        return text.replace("*", "·")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(display_math(item) for item in value) + "]"
    return str(value)


def latex_math(value: Any) -> str:
    """Return LaTeX for frontend math rendering (KaTeX/MathJax)."""
    if isinstance(value, str):
        return r"\text{" + value.replace("_", r"\_") + "}"
    try:
        return sp.latex(value)
    except (TypeError, ValueError):
        return r"\text{" + display_math(value).replace("_", r"\_") + "}"


def inline_math(value: Any) -> str:
    return f"${latex_math(value)}$"


def pack(operation: str, result: Any, steps: list[str],
         verified: bool | None = True) -> dict:
    shown_result = display_math(result)
    response = f"**{operation}**\n\n**Result:** {inline_math(result)}\n\n"
    if steps:
        response += "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(steps)
        ) + "\n\n"
    response += f"**Verification:** {'PASSED' if verified else 'NOT CONFIRMED'}"
    return {
        "success": bool(verified), "operation": operation, "result": shown_result,
        "steps": steps, "verification": bool(verified) if verified is not None else None,
        "response": response,
    }

def execute(p: dict) -> dict:
    category, operation = p.get("category"), p.get("operation")
    if (category, operation) not in OPS:
        return {
            "success": False, "operation": None, "result": None, "steps": [],
            "verification": None,
            "response": p.get("clarification") or
                        "Please rephrase your mathematical question and include the required expression or values.",
        }

    if category == "equation":
        raw = str(p.get("expression", "")).strip()
        if not raw:
            raise ValueError("Please provide an equation or expression.")
        if raw.count("=") > 1:
            raise ValueError("Please provide an equation with only one equals sign.")
        if "=" in raw:
            left, right = raw.split("=", 1)
            left_expr, right_expr = expr(left), expr(right)
            equation_expr = left_expr - right_expr
            equation = sp.Eq(left_expr, right_expr)
        else:
            equation_expr = expr(raw)
            equation = sp.Eq(equation_expr, 0)
        free_symbols = equation_expr.free_symbols
        solve_variable = next(iter(free_symbols)) if len(free_symbols) == 1 else X
        solutions = sp.solve(equation, solve_variable)
        verified = all(sp.simplify(equation_expr.subs(solve_variable, value)) == 0 for value in solutions)
        steps = [
            f"Original equation: {inline_math(equation)}.",
            f"Rearrange to zero: {inline_math(sp.Eq(sp.expand(equation_expr), 0))}.",
            f"Solution set: {inline_math(sp.FiniteSet(*solutions))}.",
        ]
        return pack("Equation solving", solutions, steps, verified)

    if category == "calculus":
        variable = SYMBOLS.get(str(p.get("variable", "x")), X)
        expression = expr(str(p.get("expression", "")))
        if operation == "derivative":
            result = sp.diff(expression, variable)
            steps = [f"Given: {inline_math(sp.Function(chr(102))(variable))} = {inline_math(expression)}.",
                     f"Differentiate with respect to {variable} using the applicable derivative rules.",
                     f"Therefore, {inline_math(sp.diff(expression, variable))}."]
            return pack("Derivative", result, steps,
                        sp.simplify(sp.diff(expression, variable) - result) == 0)
        if operation == "partial_derivative":
            result = sp.diff(expression, variable)
            return pack("Partial derivative", result, [
                f"Differentiate with respect to {variable}, treating other variables as constants.",
                f"Therefore, {inline_math(result)}.",
            ], True)
        if operation == "integral":
            if p.get("lower") is not None and p.get("upper") is not None:
                lower = expr(str(p["lower"]))
                upper = expr(str(p["upper"]))
                result = sp.integrate(expression, (variable, lower, upper))
                antiderivative = sp.integrate(expression, variable)
                verified = sp.simplify(
                    sp.diff(antiderivative, variable) - expression
                ) == 0
                return pack("Definite integral", result, [
                    f"Integrand: {inline_math(expression)}.",
                    f"Bounds: {inline_math(lower)} to {inline_math(upper)}.",
                    "Evaluate the antiderivative at the upper and lower bounds.",
                    f"Result: {inline_math(result)}.",
                ], verified)
            antiderivative = sp.integrate(expression, variable)
            integration_constant = sp.Symbol("C")
            result = antiderivative + integration_constant
            verified = sp.simplify(sp.diff(antiderivative, variable) - expression) == 0
            term_steps = []
            for term in sp.Add.make_args(expression):
                term_integral = sp.integrate(term, variable)
                term_steps.append(
                    f"{inline_math(sp.Integral(term, variable))} = {inline_math(term_integral)}"
                )
            steps = [
                f"Start with: {inline_math(sp.Integral(expression, variable))}.",
                "Use linearity to integrate each term separately:",
                *term_steps,
                f"Combine the antiderivatives and add the constant: {inline_math(result)}.",
            ]
            packed = pack("Integral", result, steps, verified)
            # Keep the integration constant at the end in the user-facing LaTeX.
            ordered_latex = sp.latex(antiderivative) + r" + C"
            packed["result"] = display_math(antiderivative) + " + C"
            packed["response"] = packed["response"].replace(
                inline_math(result), f"${ordered_latex}$"
            )
            return packed
        if operation == "limit":
            point = p.get("point")
            if point is None:
                raise ValueError("Please specify the value that the variable approaches.")
            result = sp.limit(expression, variable, expr(str(point)))
            return pack("Limit", result, [
                f"Evaluate the limit as {variable} approaches {point}.",
                f"Result: {display_math(result)}",
            ], True)

    if category == "algebra":
        expression = expr(str(p.get("expression", "")))
        funcs = {
            "simplify": sp.simplify, "factor": sp.factor,
            "expand": sp.expand, "evaluate": lambda value: sp.simplify(sp.N(value)),
        }
        result = funcs[operation](expression)
        return pack(operation.capitalize(), result, [
            f"Input: {display_math(expression)}", f"Apply {operation}.", f"Result: {display_math(result)}",
        ], True)

    if category == "matrix":
        if operation in ("multiply", "add", "subtract"):
            a, b = mat(p.get("matrix_a")), mat(p.get("matrix_b"))
            if operation == "multiply":
                if a.cols != b.rows:
                    raise ValueError("Matrix dimensions are incompatible for multiplication.")
                result = a * b
            else:
                if a.shape != b.shape:
                    raise ValueError("Matrices must have the same dimensions.")
                result = a + b if operation == "add" else a - b
            return pack(f"Matrix {operation}", result, [
                f"A =\n{a}", f"B =\n{b}", f"Result =\n{result}",
            ], True)

        a = mat(p.get("matrix"))
        if operation in ("determinant", "inverse", "eigenvalues") and a.rows != a.cols:
            raise ValueError(f"{operation.capitalize()} requires a square matrix.")
        if operation == "determinant":
            result = a.det()
        elif operation == "inverse":
            if a.det() == 0:
                raise ValueError("This matrix is singular and has no inverse.")
            result = a.inv()
        elif operation == "transpose":
            result = a.T
        elif operation == "rank":
            result = a.rank()
        else:
            result = a.eigenvals()
        verified = (a * result == sp.eye(a.rows)) if operation == "inverse" else True
        return pack(f"Matrix {operation}", result, [
            f"Given matrix:\n{a}", f"Result:\n{display_math(result)}",
        ], verified)

    if category == "statistics":
        values = p.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("Please provide a list of numbers.")
        values = [
            sp.Rational(str(value)) if isinstance(value, (int, float, str)) else sp.sympify(value)
            for value in values
        ]
        mean_value = sp.Rational(sum(values), len(values))
        if operation == "mean":
            result = mean_value
        elif operation == "median":
            ordered = sorted(values, key=float)
            middle = len(ordered) // 2
            result = ordered[middle] if len(ordered) % 2 else sp.Rational(
                ordered[middle - 1] + ordered[middle], 2
            )
        elif operation == "mode":
            from collections import Counter
            counts = Counter(values)
            highest = max(counts.values())
            result = sorted(
                [value for value, count in counts.items() if count == highest],
                key=str,
            )
        else:
            variance = sp.Rational(
                sum((value - mean_value) ** 2 for value in values), len(values)
            )
            result = variance if operation == "variance" else sp.sqrt(variance)
        return pack(operation.replace("_", " ").capitalize(), result, [
            f"Data: {values}", f"Result: {display_math(result)}",
        ], True)

    if category == "number_theory":
        if operation == "prime_factors":
            value = int(p.get("value"))
            if value < 1:
                raise ValueError("Please provide a positive integer.")
            result = sp.factorint(value)
        else:
            values = p.get("values", [])
            if len(values) < 2:
                raise ValueError("Please provide at least two integers.")
            integers = [int(value) for value in values]
            result = sp.igcd(*integers) if operation == "gcd" else sp.ilcm(*integers)
        return pack(
            operation.upper() if operation in ("gcd", "lcm") else "Prime factorization",
            result, [f"Result: {display_math(result)}"], True,
        )

    raise ValueError("This operation is not implemented yet.")


def answer_math(text: str, history: list[dict] | None = None) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False, "operation": None, "result": None, "steps": [],
            "verification": None, "response": "Please enter a mathematical question.",
        }
    try:
        return execute(plan(text.strip(), history or []))
    except (ValueError, TypeError, SyntaxError, sp.SympifyError,
            ZeroDivisionError, OverflowError) as error:
        return {
            "success": False, "operation": None, "result": None, "steps": [],
            "verification": None, "response": f"I couldn't process that request: {error}",
        }
    except Exception:
        return {
            "success": False, "operation": None, "result": None, "steps": [],
            "verification": None,
            "response": "Something went wrong while processing this calculation. Please check the expression and try again.",
        }
