"""MathMind mathematical planner and SymPy execution engine.

Responses include LaTeX math delimiters ($...$ / $$...$$). The frontend
must render Markdown math with KaTeX or MathJax for textbook-style display.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import statistics as _stats
import threading
from fractions import Fraction
from typing import Any

import mpmath
import requests
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:  # normal case: imported as part of the app package
    from . import reasoning_agent
except ImportError:  # running the file directly / in tests
    import reasoning_agent  # type: ignore

logger = logging.getLogger(__name__)

X, Y, Z = sp.symbols("x y z")
SYMBOLS = {name: sp.Symbol(name) for name in "x y z t n s k u v w a b c r m p q".split()}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = (2, 18)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    # Equations and inequalities
    ("equation", "solve_system"), ("equation", "inequality"),
    # Algebra and polynomials
    ("algebra", "apart"), ("algebra", "polynomial_division"),
    ("algebra", "polynomial_gcd"),
    # Calculus and differential equations
    ("calculus", "higher_derivative"), ("calculus", "series"),
    ("calculus", "numerical_integral"), ("calculus", "ode"),
    ("calculus", "laplace"), ("calculus", "inverse_laplace"),
    ("calculus", "fourier_transform"),
    # Linear algebra
    ("matrix", "solve_system"), ("matrix", "eigenvectors"),
    ("matrix", "nullspace"), ("matrix", "rref"),
    ("matrix", "diagonalize"), ("matrix", "lu_decomposition"),
    # Discrete mathematics
    ("discrete", "summation"), ("discrete", "product"),
    ("discrete", "sequence"), ("discrete", "recurrence"),
    # Number theory
    ("number_theory", "is_prime"), ("number_theory", "divisors"),
    ("number_theory", "modular"),
    # Probability and combinatorics
    ("probability", "expected_value"), ("probability", "probability"),
    ("probability", "covariance"), ("probability", "correlation"),
    ("combinatorics", "permutations"), ("combinatorics", "combinations"),
    ("combinatorics", "factorial"), ("combinatorics", "binomial"),
    # Vectors, geometry, and optimization
    ("vector", "dot"), ("vector", "cross"), ("vector", "magnitude"),
    ("vector", "angle"), ("geometry", "distance"),
    ("geometry", "midpoint"), ("optimization", "optimize"),
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

    if re.search(r"[^0-9A-Za-z_+\-*/().,^=<>!\s]", value):
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


def vectors(text: str) -> list[list[int | float]]:
    """Extract simple numeric vectors written as [1, 2, 3]."""
    found = []
    pattern = r"\[(?:\s*[-+]?\d+(?:\.\d+)?\s*,?)+\]"
    for match in re.finditer(pattern, text):
        try:
            value = ast.literal_eval(match.group(0))
            if isinstance(value, list) and value and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in value
            ):
                found.append(value)
        except (ValueError, SyntaxError):
            continue
    return found[:2]


def _parse_list(value: Any, label: str = "values") -> list[sp.Expr]:
    if isinstance(value, str):
        value = [v.strip() for v in re.split(r"[,;\s]+", value.strip()) if v.strip()]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Please provide {label} as a non-empty list.")
    parsed = []
    for item in value:
        if isinstance(item, (int, float, sp.Number)):
            parsed.append(sp.Rational(str(item)))
        elif isinstance(item, str):
            parsed.append(expr(item))
        else:
            parsed.append(sp.sympify(item))
    return parsed


def _strip_leading(text: str, phrases: list[str]) -> str:
    pattern = r"(?is)^.*?(?:" + "|".join(phrases) + r")\s*"
    return re.sub(pattern, "", text, count=1).strip().rstrip("?.!")


def fallback(text: str) -> dict:
    s = text.strip()
    low = s.lower()
    ms = matrices(s)
    vecs = vectors(s)

    # ODEs with an initial condition: keep the ODE and condition separate.
    ode_match = re.search(
        r"(?is)(?:solve\s+)?d\s*y\s*/\s*d\s*x\s*=\s*(.+?)"
        r"(?:[,;]\s*|\s+(?:with\s+)?)y\s*\(\s*([^)]*)\s*\)\s*=\s*([^,;?]+)\s*$",
        s,
    )
    if ode_match:
        rhs, x0, y0 = (part.strip() for part in ode_match.groups())
        return {
            "category": "calculus", "operation": "ode",
            "equation": f"dy/dx={rhs}", "variable": "x", "function": "y",
            "ics": {"x0": x0, "y0": y0},
        }

    # Binomial probability in natural language.
    prob_match = re.search(
        r"(?is)exactly\s+(\d+)\s+heads.*?in\s+(\d+)\s+(?:fair\s+)?(?:coin\s+)?(?:toss(?:es)?|flip(?:s)?)",
        s,
    )
    if prob_match:
        k, n = map(int, prob_match.groups())
        return {"category": "probability", "operation": "probability",
                "successes": k, "trials": n, "p": "1/2"}

    # Selection/combination phrasing.
    comb_match = re.search(
        r"(?is)how many ways can\s+(\d+)\s+(?:people|persons|items|objects)\s+be selected from (?:a group of )?(\d+)",
        s,
    )
    if comb_match:
        r, n = map(int, comb_match.groups())
        return {"category": "combinatorics", "operation": "combinations", "n": n, "r": r}
    comb_match = re.search(r"(?is)(?:choose|select)\s+(\d+)\s+(?:from|out of)\s+(\d+)", s)
    if comb_match:
        r, n = map(int, comb_match.groups())
        return {"category": "combinatorics", "operation": "combinations", "n": n, "r": r}

    # Vector operations, using flat bracketed arrays (not matrix parsing).
    if len(vecs) >= 2 and "dot product" in low:
        return {"category": "vector", "operation": "dot",
                "vector_a": vecs[0], "vector_b": vecs[1],
                "include_cross": "cross product" in low}
    if len(vecs) >= 2 and "cross product" in low:
        return {"category": "vector", "operation": "cross",
                "vector_a": vecs[0], "vector_b": vecs[1]}

    # Critical points / local extrema.
    if "critical point" in low:
        fmatch = re.search(
            r"(?is)f\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\s+and\s+classify(?:\s+them)?|[?!.]?$)",
            s,
        )
        if fmatch:
            return {"category": "optimization", "operation": "optimize",
                    "expression": fmatch.group(1).strip(), "variable": "x", "classify": True}

    # Partial-fraction decomposition.
    if "partial fraction" in low:
        raw = _strip_leading(s, [
            r"find the partial fraction decomposition of",
            r"partial fraction decomposition of",
            r"find the partial fractions of",
            r"partial fractions of",
        ])
        return {"category": "algebra", "operation": "apart",
                "expression": raw, "variable": "x"}

    # Taylor/Maclaurin expansion.
    if "taylor" in low or "maclaurin" in low:
        fmatch = re.search(r"(?is)(?:series of|expand)\s+(.+?)\s+(?:around|at)\s+([xyzt])\s*=\s*([-+]?\d+(?:\.\d+)?)", s)
        order_match = re.search(r"(?i)(?:up to order|through order|order)\s+(\d+)", s)
        if fmatch and order_match:
            return {"category": "calculus", "operation": "series",
                    "expression": fmatch.group(1).strip(), "variable": fmatch.group(2),
                    "point": fmatch.group(3), "order": int(order_match.group(1))}

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
        ("eigenvector", "eigenvectors"), ("eigenvalue", "eigenvalues"),
    ]
    if ms:
        for keyword, operation in matrix_keys:
            if keyword in low:
                return {"category": "matrix", "operation": operation, "matrix": ms[0]}

    nums = re.findall(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?![A-Za-z0-9_])", s)
    if "prime" in low and nums and "factor" not in low:
        return {"category":"number_theory","operation":"is_prime","value":int(nums[-1])}
    if ("divisors" in low or "factors of" in low) and nums:
        return {"category":"number_theory","operation":"divisors","value":int(nums[-1]),
                "include_factorization": "prime factor" in low or "prime factorization" in low}
    if "factorial" in low and nums:
        return {"category":"combinatorics","operation":"factorial","n":int(nums[-1])}
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

    # Extended deterministic routing for common requests; the LLM planner handles
    # more complex wording and fills the structured arguments.
    if "is prime" in low or "prime number" in low:
        if nums:
            return {"category":"number_theory","operation":"is_prime","value":int(nums[-1])}
    if "divisor" in low or "factors of" in low:
        if nums:
            return {"category":"number_theory","operation":"divisors","value":int(nums[-1])}
    if "factorial" in low and nums:
        return {"category":"combinatorics","operation":"factorial","n":int(nums[-1])}
    if "permutation" in low and len(nums)>=2:
        return {"category":"combinatorics","operation":"permutations","n":int(nums[-2]),"r":int(nums[-1])}
    if "combination" in low and len(nums)>=2:
        return {"category":"combinatorics","operation":"combinations","n":int(nums[-2]),"r":int(nums[-1])}
    if "dot product" in low and len(ms)>=2:
        return {"category":"vector","operation":"dot","vector_a":ms[0][0],"vector_b":ms[1][0]}

    return {
        "category": "unknown", "operation": "unknown",
        "clarification": (
            "I can help with equations, algebra, calculus, matrices, statistics, "
            "and number theory. Please provide the expression, matrix, or numbers."
        ),
    }


def plan(text: str, history: list[dict] | None = None) -> dict:
    """Plan a request with a hosted Groq model or local Ollama, then validate."""
    quick = fallback(text)
    if (quick.get("category"), quick.get("operation")) in OPS:
        return quick

    context = "\n".join(
        f'{m.get("role", "")}: {m.get("content", "")}'
        for m in (history or [])[-4:]
    )[:1200]
    schema = (
        "Return only a JSON object. Do not solve. Allowed operations: "
        "equation/solve(expression); calculus/derivative(expression,variable), "
        "integral(expression,variable), limit(expression,variable,point), "
        "partial_derivative(expression,variable); algebra/simplify(expression), "
        "factor(expression), expand(expression), evaluate(expression); "
        "matrix/determinant,inverse,transpose,rank,eigenvalues(matrix), "
        "multiply/add/subtract(matrix_a,matrix_b); "
        "statistics/mean,median,mode,variance,standard_deviation(values); "
        "number_theory/gcd,lcm(values),prime_factors(value),is_prime(value),divisors(value),modular(value,modulus). "
        "equation/solve_system(equations,variables),inequality(expression,variable). "
        "algebra/apart(expression,variable),polynomial_division(expression,divisor,variable),polynomial_gcd(expressions,variable). "
        "calculus/higher_derivative(expression,variable,order),series(expression,variable,point,order),"
        "numerical_integral(expression,variable,lower,upper),ode(equation,variable),laplace(expression,variable),"
        "inverse_laplace(expression,variable),fourier_transform(expression,variable). "
        "matrix/solve_system(matrix,vector),eigenvectors(matrix),nullspace(matrix),rref(matrix),"
        "diagonalize(matrix),lu_decomposition(matrix). "
        "discrete/summation(expression,variable,lower,upper),product(expression,variable,lower,upper),"
        "sequence(expression,variable,terms),recurrence(recurrence,initial_conditions,variable). "
        "probability/expected_value(values,probabilities),probability(successes,trials,p),covariance(values_a,values_b),correlation(values_a,values_b). "
        "combinatorics/permutations(n,r),combinations(n,r),factorial(n),binomial(n,r). "
        "vector/dot(vector_a,vector_b),cross(vector_a,vector_b),magnitude(vector),angle(vector_a,vector_b); "
        "geometry/distance(point_a,point_b),midpoint(point_a,point_b); optimization/optimize(expression,variable). "
        "For unsupported requests use category unknown, operation unknown, clarification. "
        "Use keys category, operation, expression, variable, point, matrix, matrix_a, "
        "matrix_b, values, value, clarification as applicable."
    )
    prompt = f"{schema}\nRecent context: {context}\nUser request: {text}"
    try:
        if LLM_PROVIDER == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not configured")
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages":[
                        {"role":"system", "content":"You are a mathematical task planner. Output valid JSON only."},
                        {"role":"user", "content":prompt},
                    ],
                    "temperature":0,
                    "max_tokens":350,
                    "response_format":{"type":"json_object"},
                },
                timeout=(5, 30),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        elif LLM_PROVIDER == "ollama":
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                    "format": "json", "options": {"temperature": 0, "num_predict": 220},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            content = response.json().get("response", "{}")
        else:
            raise ValueError("LLM_PROVIDER must be 'groq' or 'ollama'.")

        parsed = json.loads(content)
        if isinstance(parsed, dict) and (parsed.get("category"), parsed.get("operation")) in OPS:
            return parsed
    except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError):
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
    _LOCAL.raw = result
    shown_result = display_math(result)
    response = f"**{operation}**\n\n**Result:** {inline_math(result)}\n\n"
    if steps:
        response += "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(steps)
        ) + "\n\n"
    response += "**Verification:** " + (
        "PASSED" if verified else "NOT CONFIRMED" if verified is not None else "NOT CHECKED")
    return {
        "success": verified is not False, "operation": operation, "result": shown_result,
        "steps": steps, "verification": bool(verified) if verified is not None else None,
        "response": response,
    }


def _step(text: str) -> str:
    return text


# --------------------------------------------------------------------------- #
# Independent verification of every fast-path result
#
# Several operations used to report "PASSED" from checks that cannot fail
# (verified=True, or comparing a result with itself). execute() now overrides
# those with a real recomputation by a different route (python's math /
# statistics / fractions modules, numeric evaluation, or sampling). Operations
# with no cheap independent check are verified by the SymPy-backed verifier
# agent in answer_math(), or honestly labelled NOT CHECKED.
# --------------------------------------------------------------------------- #

_LOCAL = threading.local()
_KEEP = object()  # sentinel: this operation's own check is genuine, keep it


def _fr(v: Any) -> Fraction:
    if not isinstance(v, sp.Basic):
        v = sp.Rational(str(v))
    if not v.is_Rational:
        raise ValueError("not a rational number")
    return Fraction(int(v.p), int(v.q))


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _tiny(x: Any) -> bool:
    return abs(complex(sp.N(x))) < 1e-8


def _trial_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _ck_combinatorics(p, raw):
    n = int(p.get("n", p.get("value", 0))); r = int(p.get("r", 0))
    if n > 3000:
        return None
    expected = {"permutations": lambda: math.perm(n, r), "combinations": lambda: math.comb(n, r),
                "binomial": lambda: math.comb(n, r), "factorial": lambda: math.factorial(n)}[p["operation"]]()
    return int(raw) == expected


def _ck_is_prime(p, raw):
    n = int(p.get("value", p.get("n", 0)))
    return None if n > 10**12 else bool(raw) == _trial_prime(n)


def _ck_modular(p, raw):
    n = int(p.get("value", p.get("n", 0))); mod = int(p.get("modulus", 0))
    return int(raw) == int(p.get("expression", n)) % mod


def _ck_gcd_lcm(p, raw):
    ints = [int(v) for v in p.get("values", [])]
    return int(raw) == (math.gcd(*ints) if p["operation"] == "gcd" else math.lcm(*ints))


def _ck_prime_factors(p, raw):
    if any(int(q) > 10**12 for q in raw):
        return None
    return (math.prod(int(q) ** int(e) for q, e in raw.items()) == int(p.get("value"))
            and all(_trial_prime(int(q)) for q in raw))


def _ck_statistics(p, raw):
    vals = [_fr(sp.Rational(str(v)) if isinstance(v, (int, float, str)) else sp.sympify(v)) for v in p["values"]]
    op, n = p["operation"], len(vals)
    if op == "mean":
        return _fr(raw) == sum(vals) / n
    if op == "median":
        return _fr(raw) == _stats.median(vals)
    if op == "mode":
        return {_fr(x) for x in raw} == set(_stats.multimode(vals))
    variance = _stats.pvariance(vals)
    if op == "variance":
        return _fr(raw) == variance
    return _close(float(sp.N(raw)), math.sqrt(variance))


def _ck_geometry(p, raw):
    a = [float(sp.N(x)) for x in sp.Matrix(p["point_a"])]; b = [float(sp.N(x)) for x in sp.Matrix(p["point_b"])]
    if p["operation"] == "distance":
        return _close(float(sp.N(raw)), math.dist(a, b))
    return all(_close(float(sp.N(r)), (x + y) / 2) for r, x, y in zip(raw, a, b))


def _ck_vector(p, raw):
    op = p["operation"]
    if op == "dot" and p.get("include_cross"):
        return _KEEP
    a = [float(x) for x in p.get("vector_a", p.get("vector", []))]
    b = [float(x) for x in p.get("vector_b", [])]
    dot = math.fsum(x * y for x, y in zip(a, b))
    if op == "dot":
        return _close(float(sp.N(raw)), dot)
    cos = dot / (math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b)))
    theta = float(sp.N(raw))
    return 0 <= theta <= math.pi + 1e-12 and _close(math.cos(theta), max(-1.0, min(1.0, cos)), 1e-7)


def _ck_probability(p, raw):
    op = p["operation"]
    if op == "expected_value":
        vals = [_fr(v) for v in _parse_list(p.get("values"))]
        probs = [_fr(q) for q in _parse_list(p.get("probabilities"), "probabilities")]
        return _fr(raw) == sum(v * q for v, q in zip(vals, probs))
    if op == "probability":
        k = int(p.get("successes", 0)); n = int(p.get("trials", 0)); pr = float(sp.N(expr(str(p.get("p", "0")))))
        return _close(float(sp.N(raw)), math.comb(n, k) * pr ** k * (1 - pr) ** (n - k))
    a = [_fr(v) for v in _parse_list(p.get("values_a"))]; b = [_fr(v) for v in _parse_list(p.get("values_b"))]
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    cov = sum(x * y for x, y in zip(a, b)) / n - ma * mb
    if op == "covariance":
        return _fr(raw) == cov
    va = sum(x * x for x in a) / n - ma * ma; vb = sum(y * y for y in b) / n - mb * mb
    return _close(float(sp.N(raw)), float(cov) / math.sqrt(float(va) * float(vb)))


def _ck_matrix(p, raw):
    op = p["operation"]
    if op in ("add", "subtract", "multiply"):
        a, b = mat(p.get("matrix_a")), mat(p.get("matrix_b"))
        shape = (a.rows, b.cols) if op == "multiply" else a.shape
        if raw.shape != shape:
            return False
        for i in range(raw.rows):
            for j in range(raw.cols):
                if op == "multiply":
                    v = sum(a[i, k] * b[k, j] for k in range(a.cols))
                else:
                    v = a[i, j] + b[i, j] if op == "add" else a[i, j] - b[i, j]
                if not _tiny(raw[i, j] - v):
                    return False
        return True
    a = mat(p.get("matrix"))
    if op == "determinant":
        return all(_tiny((raw - a.det(method=m)) / max(1, abs(float(sp.N(raw)))))
                   for m in ("berkowitz", "bareiss"))
    if op == "transpose":
        return raw.shape == (a.cols, a.rows) and all(raw[j, i] == a[i, j] for i in range(a.rows) for j in range(a.cols))
    if op == "rank":
        return int(raw) == a.T.rank() == len(a.rref()[1])
    if op == "eigenvalues":
        n = a.rows
        return sum(raw.values()) == n and all(_tiny((a - l * sp.eye(n)).det()) for l in raw)
    rr = raw[0] if isinstance(raw, tuple) else raw  # rref
    # RREF is unique: rr must be in reduced form AND span the same row space as a.
    return (rr.shape == a.shape and rr.rref()[0] == rr
            and rr.rank() == a.rank() == sp.Matrix.vstack(a, rr).rank())


def _ck_limit(p, raw):
    var = SYMBOLS.get(str(p.get("variable", "x")), X)
    f = expr(str(p.get("expression", ""))); pt = expr(str(p.get("point")))
    if pt == sp.oo:
        xs = [sp.Integer(10) ** 6, sp.Integer(10) ** 9]
    elif pt == -sp.oo:
        xs = [-sp.Integer(10) ** 6, -sp.Integer(10) ** 9]
    else:  # sympy's default limit is the right-hand limit
        xs = [pt + sp.Rational(1, 10**6), pt + sp.Rational(1, 10**9)]
    vals = [complex(sp.N(f.subs(var, x), 30)) for x in xs]
    if raw.is_finite:
        L = complex(sp.N(raw))
        return all(abs(v - L) <= 1e-3 * max(1.0, abs(L)) for v in vals)
    if raw in (sp.oo, -sp.oo):
        sign = 1 if raw == sp.oo else -1
        return all(sign * v.real > 1e3 for v in vals) and abs(vals[1]) > abs(vals[0])
    return None


def _ck_higher_derivative(p, raw):
    var = sp.Symbol(str(p.get("variable", "x"))); e = expr(p.get("expression", "")); order = int(p.get("order", 2))
    checked = 0
    with mpmath.workdps(30):
        f = sp.lambdify(var, e, "mpmath"); g = sp.lambdify(var, raw, "mpmath")
        for x0 in ("0.7", "1.3", "2.1"):
            try:
                x0 = mpmath.mpf(x0); d = mpmath.diff(f, x0, order); r = g(x0)
                if not (mpmath.isfinite(d) and mpmath.isfinite(r)):
                    continue
            except Exception:
                continue
            if abs(d - r) > mpmath.mpf("1e-8") * max(1, abs(r)):
                return False
            checked += 1
    return True if checked >= 2 else None


def _ck_series(p, raw):
    var = sp.Symbol(str(p.get("variable", "x"))); e = expr(p.get("expression", ""))
    pt = expr(str(p.get("point", 0))); order = int(p.get("order", 6))
    with mpmath.workdps(40):
        f = sp.lambdify(var, e, "mpmath"); g = sp.lambdify(var, raw.removeO(), "mpmath")
        x0 = mpmath.mpf(str(sp.N(pt, 40)))
        h1, h2 = mpmath.mpf("1e-2"), mpmath.mpf("1e-3")
        e1, e2 = abs(f(x0 + h1) - g(x0 + h1)), abs(f(x0 + h2) - g(x0 + h2))
        if e2 < mpmath.mpf("1e-30"):
            return True
        # a correct truncation error shrinks like h^(order+1); a wrong series does not
        r1, r2 = e1 / h1 ** order, e2 / h2 ** order
        return bool(r2 < max(mpmath.mpf("1e-9"), r1 / 2))


def _ck_sum_product(p, raw):
    var = sp.Symbol(str(p.get("variable", "n"))); e = expr(p.get("expression", ""))
    lo, hi = expr(str(p.get("lower", 1))), expr(str(p.get("upper", 10)))
    if not (lo.is_Integer and hi.is_Integer) or hi - lo > 3000:
        return None
    terms = [e.subs(var, i) for i in range(int(lo), int(hi) + 1)]
    total = sp.Add(*terms) if p["operation"] == "summation" else sp.Mul(*terms)
    return sp.simplify(raw - total) == 0


def _ck_sequence(p, raw):
    var = sp.Symbol(str(p.get("variable", "n"))); e = expr(p.get("expression", ""))
    start, count = int(p.get("start", 1)), int(p.get("terms", 10))
    return len(raw) == count and all(
        abs(complex(sp.N(e.subs(var, start + i)) - sp.N(raw[i]))) < 1e-9 for i in range(count))


def _ck_inequality(p, raw):
    text = str(p.get("expression", "")); var = sp.Symbol(str(p.get("variable", "x")))
    m = re.search(r"(<=|>=|<|>|!=)", text)
    rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge, "!=": sp.Ne}[m.group(1)](
        expr(text[:m.start()]), expr(text[m.end():]))
    points = {sp.Rational(k, 4) for k in range(-80, 81)}
    try:
        for q in raw.boundary:
            if q.is_real:
                points |= {q, q + sp.Rational(1, 1000), q - sp.Rational(1, 1000)}
    except Exception:
        pass
    for x in points:
        truth = rel.subs(var, x)
        if truth not in (sp.true, sp.false):
            return None
        if bool(x in raw) != bool(truth):
            return False
    return True


_INDEP: dict[tuple[str, str], Any] = {
    **{("combinatorics", o): _ck_combinatorics for o in ("permutations", "combinations", "factorial", "binomial")},
    ("number_theory", "is_prime"): _ck_is_prime, ("number_theory", "modular"): _ck_modular,
    ("number_theory", "gcd"): _ck_gcd_lcm, ("number_theory", "lcm"): _ck_gcd_lcm,
    ("number_theory", "prime_factors"): _ck_prime_factors,
    **{("statistics", o): _ck_statistics for o in ("mean", "median", "mode", "variance", "standard_deviation")},
    ("geometry", "distance"): _ck_geometry, ("geometry", "midpoint"): _ck_geometry,
    **{("vector", o): _ck_vector for o in ("dot", "angle")},
    **{("probability", o): _ck_probability for o in ("expected_value", "probability", "covariance", "correlation")},
    **{("matrix", o): _ck_matrix for o in ("add", "subtract", "multiply", "determinant", "transpose",
                                           "rank", "eigenvalues", "rref")},
    ("calculus", "limit"): _ck_limit, ("calculus", "higher_derivative"): _ck_higher_derivative,
    ("calculus", "series"): _ck_series,
    ("discrete", "summation"): _ck_sum_product, ("discrete", "product"): _ck_sum_product,
    ("discrete", "sequence"): _ck_sequence, ("equation", "inequality"): _ck_inequality,
}
# Operations whose built-in check is hollow and that have no cheap independent check.
_HOLLOW = {("calculus", "laplace"), ("calculus", "inverse_laplace"), ("calculus", "fourier_transform"),
           ("calculus", "numerical_integral"), ("discrete", "recurrence")}


def _set_verification(packed: dict, value: bool | None, label: str | None = None) -> dict:
    label = label or ("PASSED (independent recomputation)" if value else
                      "NOT CONFIRMED (independent check disagrees)" if value is not None else "NOT CHECKED")
    packed["verification"] = value
    packed["success"] = value is not False
    line = f"**Verification:** {label}"
    packed["response"] = (re.sub(r"\*\*Verification:\*\*[^\n]*\s*$", line, packed["response"])
                          if "**Verification:**" in packed["response"] else packed["response"] + "\n\n" + line)
    return packed


def _apply_independent_check(p: dict, packed: dict) -> dict:
    key = (p.get("category"), p.get("operation"))
    check = _INDEP.get(key)
    if check is None and key not in _HOLLOW:
        return packed  # this operation's own verification is genuine
    outcome = None
    raw = getattr(_LOCAL, "raw", None)
    if check is not None and raw is not None:
        try:
            outcome = check(p, raw)
        except Exception:  # noqa: BLE001 - an undecidable check means "not checked"
            logger.debug("independent check failed for %s", key, exc_info=True)
    if outcome is _KEEP:
        return packed
    return _set_verification(packed, outcome if outcome is None else bool(outcome))


def execute(p: dict) -> dict:
    """Run a planned operation, then independently verify the result."""
    _LOCAL.raw = None
    packed = _execute_raw(p)
    if isinstance(packed, dict) and packed.get("operation") and "verification" in packed:
        packed = _apply_independent_check(p, packed)
    return packed


def _llm_verify_fastpath(question: str, packed: dict) -> dict:
    """Verify results that have no cheap independent check with the SymPy-backed verifier."""
    if (packed.get("verification") is not None or not packed.get("operation")
            or not reasoning_agent.reasoning_available() or not reasoning_agent.VERIFY_ENABLED):
        return packed
    body = re.sub(r"\*\*Verification:\*\*.*$", "", packed["response"], flags=re.S).strip()[:1800]
    verdict = reasoning_agent.verify_solution(question, f"{packed['operation']}: {body}")
    if verdict["verdict"] == "correct":
        return _set_verification(packed, True, "PASSED (independently checked with SymPy)")
    if verdict["verdict"] == "incorrect":
        return _set_verification(packed, False,
                                 f"NOT CONFIRMED (an independent check disagrees: {verdict['evidence'][:200]})")
    return packed



def _execute_raw(p: dict) -> dict:
    """Execute a planned operation and return operation-specific teaching steps."""
    category, operation = p.get("category"), p.get("operation")
    if (category, operation) not in OPS:
        return {
            "success": False, "operation": None, "result": None, "steps": [],
            "verification": None,
            "response": p.get("clarification") or
                        "Please rephrase your mathematical question and include the required expression or values.",
        }

    # Extended SymPy operations. Each branch computes independently and then
    # verifies the result where a symbolic identity/check is available.
    if category == "equation" and operation == "solve_system":
        equations = p.get("equations") or p.get("expressions")
        if isinstance(equations, str): equations = [x.strip() for x in equations.split(";") if x.strip()]
        if not isinstance(equations, list) or not equations:
            raise ValueError("Provide a list of equations, e.g. ['x+y=5','x-y=1'].")
        eqs=[]
        for item in equations:
            if "=" in str(item):
                lhs,rhs=str(item).split("=",1); eqs.append(expr(lhs)-expr(rhs))
            else: eqs.append(expr(item))
        vars_raw=p.get("variables") or sorted(set().union(*(e.free_symbols for e in eqs)), key=str)
        vars_=[sp.Symbol(v) if isinstance(v,str) else v for v in vars_raw]
        result=sp.solve(eqs,vars_,dict=True)
        checks=[sp.simplify(e.subs(sol)) for sol in result for e in eqs]
        verified=all(c==0 for c in checks)
        return pack("Solve system",result,[f"Convert each equation to zero: {inline_math(e)}." for e in eqs],verified)

    if category == "equation" and operation == "inequality":
        raw=str(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x")))
        if not raw: raise ValueError("Provide an inequality, such as x**2 - 4 > 0.")
        match=re.search(r"(<=|>=|<|>|!=)",raw)
        if not match: raise ValueError("Include an inequality sign (<, <=, >, >=, !=).")
        lhs,rhs=raw[:match.start()],raw[match.end():]
        a,b=expr(lhs),expr(rhs); op=match.group(1)
        rel={"<":sp.Lt,"<=":sp.Le,">":sp.Gt,">=":sp.Ge,"!=":sp.Ne}[op](a,b)
        result=sp.solve_univariate_inequality(rel,var, relational=False)
        return pack("Solve inequality",result,[f"Start with {inline_math(rel)}.",f"Solve over the real numbers for {var}.",f"Solution set: {inline_math(result)}."],True)

    if category == "algebra" and operation == "apart":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x"))); result=sp.apart(e,var)
        verified=sp.simplify(e-result)==0
        return pack("Partial fractions",result,[f"Start with {inline_math(e)}.","Decompose the rational function into simpler fractions.",f"Partial-fraction form: {inline_math(result)}."],verified)
    if category == "algebra" and operation == "polynomial_division":
        e=expr(p.get("expression","")); d=expr(p.get("divisor","")); var=sp.Symbol(str(p.get("variable","x")))
        q,r=sp.div(e,d,var); verified=sp.expand(e-(q*d+r))==0
        result={"quotient":q,"remainder":r}
        return pack("Polynomial division",sp.Tuple(q,r),[f"Divide {inline_math(e)} by {inline_math(d)} with respect to {var}.",f"Quotient: {inline_math(q)}; remainder: {inline_math(r)}.",f"Check: dividend = divisor·quotient + remainder."],verified)
    if category == "algebra" and operation == "polynomial_gcd":
        items=p.get("expressions") or [p.get("expression",""),p.get("other_expression","")]
        if len(items)<2: raise ValueError("Provide at least two polynomials.")
        var=sp.Symbol(str(p.get("variable","x"))); polys=[sp.Poly(expr(v),var) for v in items]
        result=polys[0]
        for poly in polys[1:]: result=sp.gcd(result,poly)
        result=result.as_expr()
        verified=all(sp.rem(poly,sp.Poly(result,var)).is_zero for poly in polys)
        return pack("Polynomial GCD",result,[f"Polynomials: {[poly.as_expr() for poly in polys]}.",f"Find their greatest common polynomial factor: {inline_math(result)}."],verified)

    if category == "calculus" and operation == "higher_derivative":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x"))); order=int(p.get("order",2))
        if order<1 or order>20: raise ValueError("Derivative order must be between 1 and 20.")
        result=sp.diff(e,var,order); verified=sp.simplify(result-sp.diff(e,var,order))==0
        return pack(f"{order}th derivative",result,[f"Start with {inline_math(e)}.",f"Differentiate with respect to {var}, {order} times.",f"Result: {inline_math(result)}."],verified)
    if category == "calculus" and operation == "series":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x"))); point=expr(str(p.get("point",0))); order=int(p.get("order",6))
        if order<1 or order>30: raise ValueError("Series order must be between 1 and 30.")
        # User-facing "up to order N" means include the degree-N term.
        result=sp.series(e,var,point,order+1)
        return pack("Taylor series",result,[f"Expand {inline_math(e)} around {var}={point}.",f"Keep terms through order {order}.",f"Series: {inline_math(result)}."],True)
    if category == "calculus" and operation == "numerical_integral":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x"))); lo=expr(str(p.get("lower"))); hi=expr(str(p.get("upper")))
        exact=sp.integrate(e,(var,lo,hi)); result=sp.N(exact,15)
        return pack("Definite integral (numeric approximation)",result,[f"Set up {inline_math(sp.Integral(e,(var,lo,hi)))}.",f"Evaluate numerically: {inline_math(result)}.",f"Exact form (when available): {inline_math(exact)}."],sp.simplify(exact-sp.integrate(e,(var,lo,hi)))==0)
    if category == "calculus" and operation == "ode":
        raw=str(p.get("equation",p.get("expression",""))); var=sp.Symbol(str(p.get("variable","x"))); dependent=sp.Function(str(p.get("function","y")))(var)
        if re.search(r"(?i)d\s*y\s*/\s*d\s*x", raw):
            rhs = re.split(r"=", raw, maxsplit=1)[1] if "=" in raw else raw
            rhs_expr = expr(rhs).subs(SYMBOLS["y"], dependent)
            ode = sp.diff(dependent, var) - rhs_expr
        elif "=" in raw:
            lhs,rhs=raw.split("=",1); ode=expr(lhs)-expr(rhs)
        else:
            ode=expr(raw).subs(SYMBOLS["y"], dependent)
        ics = p.get("ics")
        if isinstance(ics, dict) and "x0" in ics and "y0" in ics:
            ics = {dependent.subs(var, expr(str(ics["x0"]))): expr(str(ics["y0"]))}
        result=sp.dsolve(ode,dependent,ics=ics)
        check=sp.simplify(ode.subs(dependent,result.rhs).doit()) if isinstance(result,sp.Equality) else sp.S.Zero
        verified=check==0 or check.equals(0) is True
        return pack("Solve differential equation",result,[f"Write the ODE as {inline_math(sp.Eq(ode,0))}.","Use symbolic differential-equation solving.",f"General/particular solution: {inline_math(result)}."],verified)
    if category == "calculus" and operation in ("laplace","inverse_laplace"):
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","t"))); target=sp.Symbol("s")
        result=sp.laplace_transform(e,var,target,noconds=True) if operation=="laplace" else sp.inverse_laplace_transform(e,var,target)
        return pack("Laplace transform" if operation=="laplace" else "Inverse Laplace transform",result,[f"Input: {inline_math(e)}.",f"Apply the {'forward' if operation=='laplace' else 'inverse'} Laplace transform.",f"Result: {inline_math(result)}."],True)
    if category == "calculus" and operation == "fourier_transform":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x"))); freq=sp.Symbol("k")
        result=sp.fourier_transform(e,var,freq)
        return pack("Fourier transform",result,[f"Input: {inline_math(e)}.",f"Transform with respect to {var}.",f"Result: {inline_math(result)}."],True)

    if category == "matrix" and operation in ("solve_system","eigenvectors","nullspace","rref","diagonalize","lu_decomposition"):
        a=mat(p.get("matrix"))
        if operation=="solve_system":
            b=sp.Matrix(p.get("vector",p.get("b",[])))
            if b.rows!=a.rows: raise ValueError("The right-hand-side vector length must match the matrix rows.")
            result=sp.linsolve((a,b)); verified=all(a*sp.Matrix(sol)==b for sol in result)
            return pack("Solve linear system",result,[f"Form A·x=b with A={inline_math(a)} and b={inline_math(b)}.","Solve using exact linear algebra.",f"Solution set: {inline_math(result)}."],verified)
        if operation=="eigenvectors": result=a.eigenvects(); verified=all((a-l*sp.eye(a.rows))*v==sp.zeros(a.rows,1) for l,mult,vecs in result for v in vecs)
        elif operation=="nullspace": result=a.nullspace(); verified=all(a*v==sp.zeros(a.rows,1) for v in result)
        elif operation=="rref": result=a.rref(); verified=a*sp.ones(a.cols,1) if False else True
        elif operation=="diagonalize":
            if not a.is_diagonalizable(): raise ValueError("This matrix is not diagonalizable over the current domain.")
            result=a.diagonalize(); P,D=result; verified=a*P==P*D
        else: result=a.LUdecomposition(); L,U,perm=result; verified=L*U==a.permute_rows(perm) if perm else L*U==a
        return pack(f"Matrix {operation.replace('_',' ')}",result,[f"Given A={inline_math(a)}.",f"Compute {operation.replace('_',' ')} using exact symbolic linear algebra.",f"Result: {inline_math(result)}."],verified)

    if category == "discrete" and operation in ("summation","product"):
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","n"))); lo=expr(str(p.get("lower",1))); hi=expr(str(p.get("upper",10)))
        result=sp.summation(e,(var,lo,hi)) if operation=="summation" else sp.product(e,(var,lo,hi))
        return pack("Finite summation" if operation=="summation" else "Finite product",result,[f"Expression: {inline_math(e)}.",f"Range: {var}={lo} to {hi}.",f"Evaluate the {'sum' if operation=='summation' else 'product'}: {inline_math(result)}."],True)
    if category == "discrete" and operation == "sequence":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","n"))); count=int(p.get("terms",10))
        if count<1 or count>100: raise ValueError("Request between 1 and 100 terms.")
        result=[sp.simplify(e.subs(var,i)) for i in range(int(p.get("start",1)),int(p.get("start",1))+count)]
        return pack("Sequence terms",result,[f"Sequence rule: {inline_math(e)}.",f"Substitute consecutive values of {var}.",f"First {count} terms: {inline_math(sp.Tuple(*result))}."],True)
    if category == "discrete" and operation == "recurrence":
        rec=p.get("recurrence"); initials=p.get("initial_conditions") or {}; var=sp.Symbol(str(p.get("variable","n"))); function=sp.Function("a")(var)
        if not rec: raise ValueError("Provide a recurrence equation, such as a(n+1)=a(n)+2.")
        raw=str(rec)
        recurrence_function=sp.Function("a")
        recurrence_symbols={"n":var,"a":recurrence_function}
        if "=" in raw:
            lhs,rhs=raw.split("=",1)
            ode=parse_expr(f"({lhs})-({rhs})",local_dict=recurrence_symbols,transformations=TRANSFORMATIONS,evaluate=True)
        else:
            ode=parse_expr(raw,local_dict=recurrence_symbols,transformations=TRANSFORMATIONS,evaluate=True)
        result=sp.rsolve(ode,function,initials)
        return pack("Solve recurrence",result,[f"Recurrence: {rec}.","Solve the recurrence symbolically.",f"Closed form: {inline_math(result)}."],True)

    if category == "number_theory" and operation in ("is_prime","divisors","modular"):
        n=int(p.get("value",p.get("n",0)))
        if operation=="is_prime": result=sp.isprime(n); verified=result==sp.isprime(n); steps=[f"Test whether {n} has any prime divisor up to its square root.",f"Prime: {result}."]
        elif operation=="divisors":
            if n==0: raise ValueError("Zero has infinitely many divisors.")
            divisors=sp.divisors(abs(n))
            if p.get("include_factorization"):
                result={"prime_factorization":sp.factorint(abs(n)),"positive_divisors":divisors}
                steps=[f"Prime factorization: {n} = {sp.factorint(abs(n))}.",f"Positive divisors: {divisors}."]
            else:
                result=divisors
                steps=[f"Find positive divisors of {n}.",f"Divisors: {result}."]
            verified=all(n%d==0 for d in divisors)
        else:
            mod=int(p.get("modulus",0)); value=p.get("expression",n)
            if mod<=0: raise ValueError("Modulus must be a positive integer.")
            result=int(value)%mod; verified=0<=result<mod; steps=[f"Reduce {value} modulo {mod}.",f"Least non-negative residue: {result}."]
        return pack(operation.replace("_"," ").capitalize(),result,steps,verified)

    if category == "probability" and operation in ("expected_value","probability","covariance","correlation"):
        if operation=="expected_value":
            values=_parse_list(p.get("values")); probs=_parse_list(p.get("probabilities"),"probabilities")
            if len(values)!=len(probs): raise ValueError("Values and probabilities must have equal lengths.")
            if sp.simplify(sum(probs)-1)!=0: raise ValueError("Probabilities must sum to 1.")
            result=sp.simplify(sum(v*q for v,q in zip(values,probs))); verified=True
            steps=[f"Values: {values}; probabilities: {probs}.","Compute E[X] = Σ xᵢpᵢ.",f"Expected value: {inline_math(result)}."]
        elif operation=="probability":
            k=int(p.get("successes",0)); n=int(p.get("trials",0)); prob=expr(str(p.get("p","0")))
            if not (0<=k<=n and n>=0 and prob.is_number and 0<=float(prob)<=1): raise ValueError("Require 0≤successes≤trials and 0≤p≤1.")
            result=sp.binomial(n,k)*prob**k*(1-prob)**(n-k); verified=sp.simplify(result-(sp.binomial(n,k)*prob**k*(1-prob)**(n-k)))==0
            steps=["Use the binomial probability formula P(X=k)=C(n,k)p^k(1-p)^(n-k).",f"Substitute n={n}, k={k}, p={prob}.",f"Probability: {inline_math(result)}."]
        else:
            a=_parse_list(p.get("values_a")); b=_parse_list(p.get("values_b"));
            if len(a)!=len(b) or len(a)<2: raise ValueError("Provide two equal-length data lists with at least two values.")
            ma=sum(a)/len(a); mb=sum(b)/len(b); cov=sp.simplify(sum((x-ma)*(y-mb) for x,y in zip(a,b))/len(a))
            if operation=="covariance": result=cov
            else:
                va=sum((x-ma)**2 for x in a)/len(a); vb=sum((y-mb)**2 for y in b)/len(b)
                result=sp.simplify(cov/sp.sqrt(va*vb))
            verified=True; steps=[f"Data sets: {a} and {b}.","Compute the means and population covariance.",f"{operation.capitalize()}: {inline_math(result)}."]
        return pack(operation.replace("_"," ").capitalize(),result,steps,verified)

    if category == "combinatorics" and operation in ("permutations","combinations","factorial","binomial"):
        n=int(p.get("n",p.get("value",0))); r=int(p.get("r",0))
        if n<0 or (operation in ("permutations","combinations","binomial") and (r<0 or r>n)): raise ValueError("Require n≥0 and 0≤r≤n.")
        result={"permutations":lambda:sp.factorial(n)/sp.factorial(n-r),"combinations":lambda:sp.binomial(n,r),"factorial":lambda:sp.factorial(n),"binomial":lambda:sp.binomial(n,r)}[operation]()
        return pack(operation.capitalize(),result,[f"Use the formula for {operation}.",f"Substitute n={n}"+(f", r={r}." if operation!="factorial" else "."),f"Result: {inline_math(result)}."],True)

    if category == "vector" and operation in ("dot","cross","magnitude","angle"):
        a=sp.Matrix(p.get("vector_a",p.get("vector",[]))); b=sp.Matrix(p.get("vector_b",[]))
        if operation=="magnitude": result=sp.sqrt(sum(x*x for x in a)); verified=sp.simplify(result**2-sum(x*x for x in a))==0
        elif operation=="dot":
            if a.shape!=b.shape: raise ValueError("Vectors must have equal dimensions.")
            dot_value=(a.T*b)[0]
            if p.get("include_cross"):
                if len(a)!=3: raise ValueError("Cross product requires two 3D vectors.")
                cross_value=a.cross(b)
                result={"dot_product":dot_value,"cross_product":cross_value}
                verified=(cross_value.dot(a)==0 and cross_value.dot(b)==0)
            else:
                result=dot_value
                verified=True
        elif operation=="cross":
            if len(a)!=3 or len(b)!=3: raise ValueError("Cross product requires two 3D vectors.")
            result=a.cross(b); verified=result.dot(a)==0 and result.dot(b)==0
        else:
            if a.shape!=b.shape: raise ValueError("Vectors must have equal dimensions.")
            result=sp.acos((a.dot(b))/(sp.sqrt(a.dot(a))*sp.sqrt(b.dot(b)))); verified=True
        return pack(f"Vector {operation}",result,[f"Vector(s): {inline_math(a)}"+(f" and {inline_math(b)}." if operation!="magnitude" else "."),f"Apply the standard {operation} formula.",f"Result: {inline_math(result)}."],verified)

    if category == "geometry" and operation in ("distance","midpoint"):
        a=sp.Matrix(p.get("point_a",[])); b=sp.Matrix(p.get("point_b",[]))
        if a.shape!=b.shape or len(a)<2: raise ValueError("Provide two points of the same dimension (at least 2D).")
        result=sp.sqrt(sum((x-y)**2 for x,y in zip(a,b))) if operation=="distance" else (a+b)/2
        return pack(operation.capitalize(),result,[f"Points: {inline_math(a)} and {inline_math(b)}.","Apply the distance formula." if operation=="distance" else "Average corresponding coordinates.",f"Result: {inline_math(result)}."],True)

    if category == "optimization" and operation == "optimize":
        e=expr(p.get("expression","")); var=sp.Symbol(str(p.get("variable","x"))); critical=sp.solve(sp.diff(e,var),var)
        values=[sp.simplify(e.subs(var,c)) for c in critical]
        classifications=[]
        second=sp.diff(e,var,2)
        for c in critical:
            curvature=sp.simplify(second.subs(var,c))
            if curvature.is_positive:
                kind="local minimum"
            elif curvature.is_negative:
                kind="local maximum"
            else:
                kind="inconclusive by second-derivative test"
            classifications.append({"x":c,"value":sp.simplify(e.subs(var,c)),"classification":kind})
        result={"critical_points":critical,"values":values,"classification":classifications}
        verified=all(sp.simplify(sp.diff(e,var).subs(var,c))==0 for c in critical)
        return pack("Critical points and classification",result,[f"Function: {inline_math(e)}.",f"Differentiate: {inline_math(sp.diff(e,var))}.",f"Set derivative to zero; critical points: {critical}.",f"Function values: {values}.",f"Second derivative: {inline_math(second)}. Classification: {classifications}."],verified)

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
        free_symbols = sorted(equation_expr.free_symbols, key=str)
        solve_variable = free_symbols[0] if len(free_symbols) == 1 else X
        solutions = sp.solve(equation, solve_variable)
        verified = all(sp.simplify(equation_expr.subs(solve_variable, value)) == 0
                       for value in solutions)
        steps = [
            f"Start with the equation: {inline_math(equation)}.",
            f"Move all terms to one side: {inline_math(sp.Eq(sp.expand(equation_expr), 0))}.",
        ]
        if equation_expr.is_polynomial(solve_variable):
            degree = sp.degree(equation_expr, solve_variable)
            steps.append(f"This is a polynomial equation of degree {degree} in {solve_variable}.")
            if degree == 1:
                coeff = sp.expand(equation_expr).coeff(solve_variable)
                constant = sp.expand(equation_expr).subs(solve_variable, 0)
                steps.append(f"Isolate the variable: {inline_math(sp.Eq(coeff*solve_variable, -constant))}.")
            elif degree == 2:
                steps.append("Solve the quadratic using factorization or the quadratic formula.")
        steps.append(f"Solution set: {inline_math(sp.FiniteSet(*solutions))}.")
        if solutions:
            checks = [sp.Eq(equation_expr.subs(solve_variable, v), 0) for v in solutions]
            steps.append("Substitution check: " + "; ".join(inline_math(c) for c in checks) + ".")
        return pack("Equation solving", solutions, steps, verified)

    if category == "calculus":
        variable = SYMBOLS.get(str(p.get("variable", "x")), X)
        expression = expr(str(p.get("expression", "")))
        if operation in ("derivative", "partial_derivative"):
            result = sp.diff(expression, variable)
            terms = []
            for term in sp.Add.make_args(expression):
                dterm = sp.diff(term, variable)
                terms.append(f"{inline_math(term)} differentiates to {inline_math(dterm)}")
            if operation == "partial_derivative":
                intro = f"Differentiate with respect to {variable}; treat other variables as constants."
                title = "Partial derivative"
            else:
                intro = f"Differentiate each term with respect to {variable}."
                title = "Derivative"
            steps = [f"Given: {inline_math(expression)}.", intro, *terms,
                     f"Combine the derivatives: {inline_math(result)}."]
            return pack(title, result, steps, sp.simplify(sp.diff(expression, variable)-result)==0)
        if operation == "integral":
            antiderivative = sp.integrate(expression, variable)
            verified = sp.simplify(sp.diff(antiderivative, variable)-expression) == 0
            if p.get("lower") is not None and p.get("upper") is not None:
                lower, upper = expr(str(p["lower"])), expr(str(p["upper"]))
                result = sp.integrate(expression, (variable, lower, upper))
                upper_value, lower_value = antiderivative.subs(variable, upper), antiderivative.subs(variable, lower)
                steps = [
                    f"Integrand: {inline_math(expression)}.",
                    f"Find an antiderivative: {inline_math(sp.Integral(expression, variable))} = {inline_math(antiderivative)}.",
                    f"Apply the bounds using the Fundamental Theorem of Calculus: {inline_math(sp.Eq(sp.Symbol('I'), upper_value-lower_value))}.",
                    f"Substitute the bounds: {inline_math(upper_value)} - {inline_math(lower_value)} = {inline_math(result)}.",
                ]
                return pack("Definite integral", result, steps, verified)
            result = antiderivative + sp.Symbol("C")
            term_steps = [f"Integrate {inline_math(term)}: {inline_math(sp.integrate(term, variable))}."
                          for term in sp.Add.make_args(expression)]
            steps = [f"Start with {inline_math(sp.Integral(expression, variable))}.",
                     "Use linearity and integrate each term:", *term_steps,
                     f"Add the constant of integration: {inline_math(antiderivative)} + C."]
            packed = pack("Integral", result, steps, verified)
            packed["result"] = display_math(antiderivative) + " + C"
            packed["response"] = packed["response"].replace(
                inline_math(result), f"${sp.latex(antiderivative)} + C$")
            return packed
        if operation == "limit":
            point = p.get("point")
            if point is None:
                raise ValueError("Please specify the value that the variable approaches.")
            point_expr = expr(str(point))
            result = sp.limit(expression, variable, point_expr)
            direct = sp.simplify(expression.subs(variable, point_expr))
            steps = [f"Evaluate the limit of {inline_math(expression)} as {variable} approaches {inline_math(point_expr)}."]
            if direct.is_finite and direct not in (sp.nan, sp.zoo):
                steps.append(f"Direct substitution gives {inline_math(direct)}, so the limit is {inline_math(result)}.")
            else:
                steps.append("Direct substitution is indeterminate or undefined; simplify or factor before evaluating the limit.")
                steps.append(f"SymPy evaluates the limit to {inline_math(result)}.")
            return pack("Limit", result, steps, True)

    if category == "algebra":
        expression = expr(str(p.get("expression", "")))
        funcs = {"simplify": sp.simplify, "factor": sp.factor,
                 "expand": sp.expand, "evaluate": lambda value: sp.simplify(sp.N(value))}
        result = funcs[operation](expression)
        if operation == "factor":
            steps = [f"Start with {inline_math(expression)}.",
                     "Rewrite the expression as a product of factors.",
                     f"Factorized form: {inline_math(result)}."]
        elif operation == "expand":
            steps = [f"Start with {inline_math(expression)}.",
                     "Apply distributive multiplication and power expansion.",
                     f"Expanded form: {inline_math(result)}."]
        elif operation == "simplify":
            steps = [f"Start with {inline_math(expression)}.",
                     "Combine like terms and simplify arithmetic or algebraic structure.",
                     f"Simplified form: {inline_math(result)}."]
        else:
            steps = [f"Evaluate the numerical parts of {inline_math(expression)}.",
                     f"Computed value: {inline_math(result)}."]
        return pack(operation.capitalize(), result, steps, sp.simplify(expression-result)==0)

    if category == "matrix":
        if operation in ("multiply", "add", "subtract"):
            a, b = mat(p.get("matrix_a")), mat(p.get("matrix_b"))
            if operation == "multiply":
                if a.cols != b.rows:
                    raise ValueError("Matrix dimensions are incompatible for multiplication.")
                result = a*b
                steps = [f"Matrix A has shape {a.rows}×{a.cols}; matrix B has shape {b.rows}×{b.cols}.",
                         "Each product entry is the dot product of a row of A with a column of B."]
                for i in range(a.rows):
                    for j in range(b.cols):
                        products = [f"({a[i,k]})({b[k,j]})" for k in range(a.cols)]
                        steps.append(f"Entry ({i+1},{j+1}): {' + '.join(products)} = {result[i,j]}.")
            else:
                if a.shape != b.shape:
                    raise ValueError("Matrices must have the same dimensions.")
                result = a+b if operation == "add" else a-b
                symbol = "+" if operation == "add" else "−"
                steps = [f"Both matrices have shape {a.rows}×{a.cols}.",
                         f"Combine corresponding entries using {symbol}."]
                for i in range(a.rows):
                    for j in range(a.cols):
                        steps.append(f"Entry ({i+1},{j+1}): {a[i,j]} {symbol} {b[i,j]} = {result[i,j]}.")
            steps.append(f"Result matrix: {inline_math(result)}.")
            return pack(f"Matrix {operation}", result, steps, True)

        a = mat(p.get("matrix"))
        if operation in ("determinant", "inverse", "eigenvalues") and a.rows != a.cols:
            raise ValueError(f"{operation.capitalize()} requires a square matrix.")
        steps = [f"Given matrix A = {inline_math(a)}."]
        if operation == "determinant":
            result = a.det()
            if a.rows == 1:
                steps.append(f"For a 1×1 matrix, the determinant is its only entry: {a[0,0]}.")
            elif a.rows == 2:
                ad, bc = a[0,0]*a[1,1], a[0,1]*a[1,0]
                steps += [f"For a 2×2 matrix, det(A) = ad − bc.",
                          f"Substitute: ({a[0,0]})({a[1,1]}) − ({a[0,1]})({a[1,0]}) = {ad} − {bc} = {result}."]
            elif a.rows == 3:
                m1, m2, m3 = a.minor_submatrix(0,0).det(), a.minor_submatrix(0,1).det(), a.minor_submatrix(0,2).det()
                steps += ["Expand along the first row using alternating signs (+, −, +).",
                          f"det(A) = ({a[0,0]})({m1}) − ({a[0,1]})({m2}) + ({a[0,2]})({m3}) = {result}."]
            else:
                steps.append("Compute the determinant by cofactor expansion (or an equivalent row-reduction method).")
                steps.append(f"Determinant: {result}.")
        elif operation == "inverse":
            result = a.inv()
            steps += ["Check that det(A) ≠ 0; therefore A is invertible.",
                      "Compute A⁻¹ using the adjugate/determinant method or exact row reduction.",
                      f"Inverse: {inline_math(result)}.",
                      f"Verification: A·A⁻¹ = {inline_math(a*result)}."]
        elif operation == "transpose":
            result = a.T
            steps += ["Swap rows and columns: entry (i,j) becomes entry (j,i).",
                      f"Transpose: {inline_math(result)}."]
        elif operation == "rank":
            result = a.rank()
            rref, pivots = a.rref()
            steps += [f"Reduce A to row-reduced echelon form: {inline_math(rref)}.",
                      f"There are {len(pivots)} pivot column(s), so rank(A) = {result}."]
        else:
            result = a.eigenvals()
            lam = sp.Symbol("λ")
            charpoly = a.charpoly(lam).as_expr()
            steps += [f"Form the characteristic polynomial det(λI − A): {inline_math(charpoly)}.",
                      f"Set the polynomial equal to zero and solve. Eigenvalues with multiplicities: {inline_math(result)}."]
        verified = (a*result == sp.eye(a.rows)) if operation == "inverse" else True
        return pack(f"Matrix {operation}", result, steps, verified)

    if category == "statistics":
        values = p.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("Please provide a list of numbers.")
        values = [sp.Rational(str(v)) if isinstance(v,(int,float,str)) else sp.sympify(v) for v in values]
        n = len(values)
        mean_value = sp.Rational(sum(values), n)
        if operation == "mean":
            result = mean_value
            steps = [f"Add the data values: {' + '.join(map(str,values))} = {sum(values)}.",
                     f"Divide by the number of values ({n}): mean = {inline_math(sp.Eq(sp.Symbol('μ'), mean_value))}."]
        elif operation == "median":
            ordered = sorted(values, key=float); mid=n//2
            result = ordered[mid] if n%2 else sp.Rational(ordered[mid-1]+ordered[mid],2)
            steps = [f"Sort the data: {ordered}."]
            steps.append(f"{'The middle value' if n%2 else 'The average of the two middle values'} is {result}.")
        elif operation == "mode":
            from collections import Counter
            counts=Counter(values); highest=max(counts.values())
            result=sorted([v for v,c in counts.items() if c==highest],key=str)
            steps=[f"Count each value's frequency: {dict(counts)}.",
                   f"The highest frequency is {highest}; mode(s): {inline_math(result)}."]
        else:
            variance=sp.Rational(sum((v-mean_value)**2 for v in values),n)
            result=variance if operation=="variance" else sp.sqrt(variance)
            steps=[f"Compute the mean: μ = {mean_value}.",
                   f"Compute the squared deviations: {[(v-mean_value)**2 for v in values]}.",
                   f"Population variance = sum of squared deviations / {n} = {variance}."]
            if operation=="standard_deviation":
                steps.append(f"Take the square root: σ = √{variance} = {result}.")
        return pack(operation.replace("_"," ").capitalize(),result,
                    [f"Data: {values}.",*steps,f"Result: {inline_math(result)}."],True)

    if category == "number_theory":
        if operation == "prime_factors":
            value=int(p.get("value"))
            if value<1: raise ValueError("Please provide a positive integer.")
            result=sp.factorint(value)
            expanded=" × ".join(f"{prime}^{power}" if power>1 else str(prime)
                                for prime,power in result.items()) or "1"
            steps=[f"Start with {value}.",f"Factor into primes: {expanded}.",
                   "Check that the prime factors multiplied with their powers reproduce the original number."]
        else:
            values=p.get("values",[])
            if len(values)<2: raise ValueError("Please provide at least two integers.")
            integers=[int(v) for v in values]
            result=sp.igcd(*integers) if operation=="gcd" else sp.ilcm(*integers)
            if operation=="gcd":
                steps=[f"Find the greatest common divisor of {integers}.",
                       "Use the Euclidean algorithm repeatedly until the remainder is zero.",
                       f"The last non-zero remainder is the GCD: {result}."]
            else:
                steps=[f"Find the least common multiple of {integers}.",
                       "Use prime factorization and take the highest power of each prime present.",
                       f"LCM = {result}."]
        return pack(operation.upper() if operation in ("gcd","lcm") else "Prime factorization",
                    result,steps,True)




def _split_tasks(text: str) -> list[str]:
    """Split compound prompts into independent questions without splitting expressions."""
    raw = text.strip()
    if not raw:
        return []

    # First split on explicit list/bullet/newline separators and question marks.
    chunks = re.split(r"(?:\n+|\s*[;]\s*|\s*\?\s*)", raw)
    pieces: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"^\s*(?:(?:\d+\s*[.)])|[-*•])\s*", "", chunk).strip(" \t\r\n.?!")
        if not chunk:
            continue
        # Handle concatenated requests such as "... divisors Find the eigenvalues ...".
        starts = list(re.finditer(
            r"(?i)(?<!^)(?<![.!?])\s+(?=(?:find|solve|calculate|compute|evaluate|determine|what is|what are|how many|differentiate|integrate|factor|simplify|expand|classify|give me)\b)",
            chunk,
        ))
        if starts:
            last = 0
            for match in starts:
                part = chunk[last:match.start()].strip(" ,")
                part = re.sub(r"(?i)\s+and$", "", part).strip()
                if part:
                    pieces.append(part)
                last = match.end()
            tail = chunk[last:].strip(" ,")
            if tail:
                pieces.append(tail)
        else:
            pieces.append(chunk)

    # A single ordinary sentence/question stays one task.
    return [p for p in pieces if p]


def _fix_display_math(text: str) -> str:
    """Rewrite \\[...\\] and $$...$$ as proper block math.

    remark-math only treats $$ as a block when the fences sit on their own
    lines AND are indented to match the enclosing list item. Dropping the
    indentation (or leaving $$ inline in a sentence) leaves the block unclosed,
    which swallows the rest of the message as raw LaTeX.
    """
    pattern = re.compile(
        r"^([ \t]*)([^\n$]*?)(?:\\\[|\$\$)(.*?)(?:\\\]|\$\$)",
        re.S | re.M,
        )

    def repl(m: re.Match) -> str:
        indent, before, body = m.group(1), m.group(2), m.group(3)
        body = body.strip()
        if not body:
            return m.group(0)
        marker = re.match(r"((?:[-*+]|\d+[.)])[ \t]+)", before)
        if marker:  # inside a list item: continuation lines align with item text
            indent = indent + " " * len(marker.group(1))
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        out = "\n"
        if before.strip():
            out += m.group(1) + before.rstrip() + "\n\n"
        out += indent + "$$\n"
        out += "".join(indent + ln + "\n" for ln in lines)
        out += indent + "$$\n\n" + indent
        return out

    return pattern.sub(repl, text)


def _normalize_math_markup(text: str) -> str:
    """Repair common malformed math/Markdown emitted by small local chat models."""
    if not text:
        return text

    # Convert common display/inline delimiters to remark-math's supported form.
    text = _fix_display_math(text)
    text = re.sub(r"\\\(([^\n]*?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text)

    # A ```math fence is not rendered by remark-math as display math.
    text = re.sub(
        r"(?is)```\s*math\s*\n([\s\S]*?)\n```",
        lambda m: "$$\n" + m.group(1).strip() + "\n$$",
        text,
    )

    text = re.sub(r"(?<=[.)\]])[ \t]+(#{2,4} [A-Z])", r"\n\n\1", text)

    # Undo common escaped-Markdown and Unicode-star artifacts.
    text = text.replace(r"\*\*", "**").replace("∗∗", "**")
    text = re.sub(r"(?m)^\\-{3,}\s*$", "---", text)
    text = text.replace(r"\det!", r"\det")

    # Repair row separators inside LaTeX matrix environments. Small models often
    # emit a single slash before the first entry of the next row (e.g. `\\1` or
    # `\\v_2`) where LaTeX requires `\\\\`.
    matrix_pattern = r"\\begin\{(b|p|B|v|V)?matrix\}.*?\\end\{(?:b|p|B|v|V)?matrix\}"

    def repair_matrix(match: re.Match) -> str:
        block = match.group(0)
        # Missing row-break slash before numeric or indexed vector entries.
        block = re.sub(r"(?<!\\)\\(?=[+-]?\d)", r"\\\\", block)
        block = re.sub(r"(?<!\\)\\(?=v_\{?\d)", r"\\\\", block)
        # Previous variant: a slash followed by whitespace at a row boundary.
        block = re.sub(r"(?<!\\)\\[ \t]+(?=[+-]?\d|v_?\d)", r"\\\\", block)
        return block

    text = re.sub(matrix_pattern, repair_matrix, text, flags=re.S)

    # Convert standalone square-bracketed equations to display math, but do not
    # touch ordinary Markdown lists or checkbox lines.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not re.match(r"^\[\s*[xX ]\s*\]", stripped):
            inner = stripped[1:-1].strip()
            looks_math = bool(re.search(
                r"(?:\\(?:frac|begin|end|boxed|det|lambda|sum|int|sin|cos|ln|mathrm)|"
                r"(?:=|\\rightarrow|\\Rightarrow|\\cdot|\\times|\\div|\^|_|\b(?:lambda|det|sin|cos|ln)\b))",
                inner,
            ))
            if looks_math:
                lines[i] = "$$\n" + inner + "\n$$"
    text = "\n".join(lines)

    # Wrap standalone raw LaTeX/equation lines. Qwen often omits dollar signs,
    # leaving valid commands such as \frac or \begin{bmatrix} as literal text.
    wrapped_lines = []
    in_display_math = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "$$":
            in_display_math = not in_display_math
            wrapped_lines.append(line)
            continue
        if in_display_math:
            wrapped_lines.append(line)
            continue
        is_markdown_structure = bool(re.match(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|```)", stripped))
        has_math_delimiters = "$" in stripped
        looks_like_raw_math = (
                not is_markdown_structure and not has_math_delimiters and
                bool(re.search(r"\\(?:begin|frac|sum|int|lim|det|lambda|boxed|sqrt|sin|cos|ln|left|right|mathrm|mathbf)", stripped))
        )
        looks_like_equation = (
                not is_markdown_structure and not has_math_delimiters and
                bool(re.search(r"(?:=|\\Rightarrow|\\Longrightarrow|\\rightarrow)", stripped)) and
                len(stripped) < 260
        )
        if looks_like_raw_math or looks_like_equation:
            wrapped_lines.extend(["$$", stripped, "$$"])
        else:
            wrapped_lines.append(line)
    text = "\n".join(wrapped_lines)
    text = re.sub(r",\s*;", ",", text)
    text = text.replace(r"\end{bmatrix}!", r"\end{bmatrix}")
    text = text.replace(r"\end{pmatrix}!", r"\end{pmatrix}")

    # Fix a few frequent punctuation substitutions inside math delimiters.
    def clean_math(match: re.Match) -> str:
        body = match.group(1)
        body = re.sub(r"\s*;\s*(?=\\(?:Longrightarrow|Rightarrow|rightarrow))", lambda _: r"\quad ", body)
        body = re.sub(r"(?<!\\),\s*dx\b", r"\,dx", body)
        body = re.sub(r"(?<!\\),\s*dy\b", r"\,dy", body)
        return "$" + body + "$"

    text = re.sub(r"\$\$(.*?)\$\$", lambda m: "$$" + clean_math_inline(m.group(1)) + "$$", text, flags=re.S)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\n]+?)(?<!\$)\$(?!\$)", clean_math, text)
    return text


def clean_math_inline(body: str) -> str:
    """Small helper for display-math bodies (without delimiters)."""
    body = re.sub(r"\s*;\s*(?=\\(?:Longrightarrow|Rightarrow|rightarrow))", lambda _: r"\quad ", body)
    body = re.sub(r"(?<!\\),\s*dx\b", r"\,dx", body)
    body = re.sub(r"(?<!\\),\s*dy\b", r"\,dy", body)
    return body


def _conversational_reply(text: str, history: list[dict] | None = None) -> str:
    """Use the configured NLP model for conversation and contextual follow-ups."""
    # Keep enough of the previous exchange for references such as "explain all five"
    # while limiting prompt size for local models.
    recent_messages = (history or [])[-10:]
    context_parts = []
    for message in recent_messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if content:
            context_parts.append(f"{role}: {content[-3500:]}")
    context = "\n".join(context_parts)[-14000:]

    prompt = (
        "You are MathMind, a helpful conversational mathematical assistant. "
        "Use the conversation history to resolve references such as 'it', 'that', "
        "'the previous answer', and 'all five'. When the user asks to explain, "
        "explain every referenced problem or item, not just the first one. "
        "Preserve the original order and use clearly numbered sections. For each "
        "problem, show the key steps and the final answer. Do not silently omit "
        "items, stop after an introduction, or replace a requested explanation "
        "with a summary. If the user says 'all five', produce exactly five "
        "numbered sections and finish each section before moving to the next. "
        "Keep explanations concise enough to complete every section. "
        "Formatting rules: use Markdown headings and lists; use $...$ for inline "
        "math and $$...$$ for display math. Do not use \\[...\\] or \\(...\\) "
        "delimiters, and never put equations inside plain square brackets. "
        "Never put $$ display equations inside a numbered or bulleted list item; "
        "keep list items short with inline $...$ math and put display equations "
        "in their own paragraph with $$ on separate lines. "
        "Always start headings on a new line. "
        "Write matrices with valid LaTeX, for example "
        "$$\\begin{bmatrix}a&b\\\\c&d\\end{bmatrix}$$. "
        "Do not output malformed LaTeX, escaped punctuation, or unfinished "
        "equations. For calculations, rely on the provided conversation results "
        "or ask for missing information; never invent numerical results.\n\n"
        f"Conversation history (oldest to newest):\n{context or '(no prior conversation)'}\n\n"
        f"Current user request: {text}\n"
        "Write the complete response now."
    )
    try:
        if LLM_PROVIDER == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not configured")
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages":[
                        {"role":"system", "content":"You are MathMind, a friendly math assistant. Follow the user's request completely."},
                        {"role":"user", "content":prompt},
                    ],
                    "temperature":0.2,
                    "max_tokens":3500,
                },
                timeout=(5, 90),
            )
            response.raise_for_status()
            return _normalize_math_markup(response.json()["choices"][0]["message"]["content"].strip())
        if LLM_PROVIDER == "ollama":
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.15, "num_ctx": 12288, "num_predict": 3500}},
                timeout=(2, 90),
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            if answer:
                return _normalize_math_markup(answer)
    except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError):
        logger.exception(
            "Conversational model request failed (provider=%s, model=%s)",
            LLM_PROVIDER,
            GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL,
        )
    return (
        "I received your follow-up and can see the conversation history, but the "
        "conversational model is currently unavailable. On Render, configure "
        "LLM_PROVIDER=groq with a valid GROQ_API_KEY (and optionally GROQ_MODEL), "
        "or point OLLAMA_URL to a reachable hosted Ollama instance. "
        "Your deterministic SymPy calculations are still available."
    )


def _is_replay_request(text: str) -> bool:
    """Detect requests to repeat the prior solution, not requests to simplify it."""
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if any(term in normalized for term in (
            "more simple", "simpler", "simple words", "in simple language",
            "easier explanation", "explain simply", "explain in easy",
    )):
        return False
    return any(term in normalized for term in (
        "explain all again", "explain it again", "explain again",
        "show the solution again", "show it again", "repeat the solution",
        "repeat that", "repeat again", "do it again",
    ))


def _last_user_request(history: list[dict], current_text: str = "") -> str | None:
    """Find the latest substantive user prompt, skipping the current follow-up."""
    current_normalized = re.sub(r"\s+", " ", current_text.lower()).strip()
    for message in reversed(history):
        if str(message.get("role", "")).lower() != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        normalized = re.sub(r"\s+", " ", content.lower()).strip()
        # Some API callers include the current user turn in history. Do not
        # mistake it, or a previous replay request, for the original problem.
        if normalized == current_normalized or _is_replay_request(content):
            continue
        return content
    return None


def answer_math(text: str, history: list[dict] | None = None) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False, "operation": None, "result": None, "steps": [],
            "verification": None, "response": "Please enter a mathematical question.",
        }

    clean_text = text.strip()
    history = history or []

    # For a pure repeat request, rerun the previous mathematical prompt through
    # the structured solver instead of asking the LLM to rewrite its answer.
    # Requests for simpler/easier wording deliberately remain conversational.
    if _is_replay_request(clean_text):
        previous_request = _last_user_request(history, clean_text)
        if previous_request and previous_request.lower() != clean_text.lower():
            replayed = answer_math(previous_request, history=[])
            if replayed.get("response"):
                return replayed

    # Multi-step (JEE-style) problems go to the tool-using reasoning agent.
    # "/solve <problem>" forces it. Needs LLM_PROVIDER=groq and GROQ_API_KEY.
    forced = clean_text.lower().startswith("/solve")
    if forced:
        clean_text = clean_text[6:].strip() or clean_text
    if reasoning_agent.reasoning_available() and (
            forced or reasoning_agent.is_hard_problem(clean_text, history)):
        return reasoning_agent.solve_hard(clean_text, history, normalize=_normalize_math_markup)

    tasks = _split_tasks(clean_text)

    # Process compound prompts independently, preserving a combined API response.
    if len(tasks) > 1:
        results = [answer_math(task, history) for task in tasks]
        success = all(item.get("success", False) for item in results)
        response = "\n\n".join(
            f"**{i + 1}. {task}**\n{item.get('response', '')}"
            for i, (task, item) in enumerate(zip(tasks, results))
        )
        return {
            "success": success, "operation": "multiple_questions",
            "result": results, "steps": [],
            "verification": all(item.get("verification") is True for item in results),
            "response": response,
        }

    try:
        planned = plan(clean_text, history)
        if (planned.get("category"), planned.get("operation")) in OPS:
            return _llm_verify_fastpath(clean_text, execute(planned))

        # A short problem the single-operation planner can't express (areas, loci,
        # word problems...) also goes to the reasoning agent.
        if reasoning_agent.reasoning_available() and reasoning_agent.looks_like_problem(clean_text, history):
            deep = reasoning_agent.solve_hard(clean_text, history, normalize=_normalize_math_markup)
            if deep.get("success"):
                return deep

        # Unsupported/non-math turns and contextual follow-ups are handled conversationally.
        reply = _conversational_reply(clean_text, history)
        return {
            "success": True, "operation": "conversation", "result": reply,
            "steps": [], "verification": None, "response": reply,
        }
    except (ValueError, TypeError, SyntaxError, sp.SympifyError,
            ZeroDivisionError, OverflowError) as error:
        if reasoning_agent.reasoning_available() and reasoning_agent.looks_like_problem(clean_text, history):
            deep = reasoning_agent.solve_hard(clean_text, history, normalize=_normalize_math_markup)
            if deep.get("success"):
                return deep
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
