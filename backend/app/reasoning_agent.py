from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
import re
import time
from typing import Any, Callable
import wolframalpha

import requests
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from . import auditor
except ImportError:
    import auditor  # type: ignore

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "high").strip().lower()
MAX_STEPS = int(os.getenv("JEE_MAX_STEPS", "12"))
SAMPLES = max(1, min(int(os.getenv("JEE_SAMPLES", "3")), 3))
DEADLINE = float(os.getenv("JEE_DEADLINE_SECONDS", "120"))
VERIFY_ENABLED = os.getenv("JEE_VERIFY", "1").strip() != "0"
MAX_FIX_ROUNDS = max(0, int(os.getenv("JEE_FIX_ROUNDS", "2")))
AUDIT_ENABLED = os.getenv("JEE_AUDIT", "1").strip() != "0"
TOOL_TIMEOUT = 12
MAX_TOOL_OUTPUT = 4000

def reasoning_available() -> bool:
    return (
            os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "groq"
            and bool(os.getenv("GROQ_API_KEY"))
    )

_FOLLOWUP = re.compile(
    r"^\s*(explain|why|how come|what do you mean|what does|simplif|elaborate|"
    r"can you explain|could you explain|in simple|again|thanks|thank you|ok\b|okay|"
    r"hi\b|hello|hey\b|summar|translate|repeat|show the solution|rewrite)",
    re.I,
)
_FOLLOWUP_ANYWHERE = re.compile(
    r"simpler words|simple words|simple language|easier|explain (it|this|that)|"
    r"why (this|that|did)|step by step explanation of (it|that)",
    re.I,
)
_MATH_CUE = re.compile(
    r"\d|[=<>^+\-*/]|\b(find|prove|show that|evaluate|determine|compute|calculate|"
    r"locus|circle|parabola|ellipse|hyperbola|tangent|normal|chord|probability|"
    r"vector|matrix|determinant|sequence|series|complex|roots?|integral|limit|"
    r"function|triangle|maximum|minimum|area|volume|angle|equation|polynomial|"
    r"permutation|combination|arithmetic|geometric|progression|trigonometric|"
    r"let|if|given|sum|product|value of|number of ways|how many)\b",
    re.I,
)


def _is_followup(text: str, history: list[dict] | None) -> bool:
    if not history:
        return False
    return bool(_FOLLOWUP.search(text) or _FOLLOWUP_ANYWHERE.search(text))


def looks_like_problem(text: str, history: list[dict] | None = None) -> bool:
    return bool(_MATH_CUE.search(text)) and not _is_followup(text, history)


def is_hard_problem(text: str, history: list[dict] | None = None) -> bool:
    words = len(text.split())
    return (len(text) >= 110 or words >= 18) and looks_like_problem(text, history)

_FUNCS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec,
    "csc": sp.csc, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "acot": sp.acot, "atan2": sp.atan2, "sinh": sp.sinh, "cosh": sp.cosh,
    "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "log10": lambda v: sp.log(v, 10), "log2": lambda v: sp.log(v, 2), "sqrt": sp.sqrt, "cbrt": sp.cbrt,
    "root": sp.root, "Abs": sp.Abs, "abs": sp.Abs, "floor": sp.floor,
    "ceiling": sp.ceiling, "factorial": sp.factorial, "binomial": sp.binomial,
    "gcd": sp.gcd, "lcm": sp.lcm, "re": sp.re, "im": sp.im,
    "conjugate": sp.conjugate, "arg": sp.arg, "sign": sp.sign,
    "Rational": sp.Rational, "Matrix": sp.Matrix, "Max": sp.Max, "Min": sp.Min,
    "pi": sp.pi, "E": sp.E, "I": sp.I, "oo": sp.oo,
}
_PARSE_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float,
    "Rational": sp.Rational, "Function": sp.Function,
}
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
_FORBIDDEN = re.compile(
    r"__|\.\s*[A-Za-z_]|[\"'`;@\\$]|\b(import|exec|eval|lambda|open|getattr|setattr|"
    r"compile|globals|locals|vars|dir|type|input|os|sys|subprocess)\b"
)


def _P(text: Any) -> Any:
    s = str(text).strip()
    if not s:
        raise ValueError("empty expression")
    if len(s) > 800:
        raise ValueError("expression too long")
    s = (s.replace("−", "-").replace("×", "*").replace("÷", "/").replace("π", "pi")
         .replace("²", "**2").replace("³", "**3").replace("√", "sqrt"))
    # x0, a23 -> x_0, a_23. Without this, implicit-multiplication splitting turns "x0" into x*0
    # (or fails to parse), so tools silently error on subscripted variables.
    s = re.sub(r"(?<![A-Za-z_0-9.])([A-Za-z])(\d+)(?![A-Za-z_(\d.])", r"\1_\2", s)
    if _FORBIDDEN.search(s):
        raise ValueError(
            "disallowed characters. Use plain syntax only: * for multiplication, "
            "** or ^ for powers, no LaTeX, no attribute access like x.is_real"
        )
    try:
        return parse_expr(
            s, local_dict=dict(_FUNCS), global_dict=dict(_PARSE_GLOBALS),
            transformations=_TRANSFORMS, evaluate=True,
        )
    except Exception as error:
        raise ValueError(f"could not parse '{s}': {error}") from error


def _eq(text: Any) -> Any:
    s = str(text).replace("==", "=")
    if "=" in s and not re.search(r"[<>!]=|=[<>]", s):
        lhs, rhs = s.split("=", 1)
        return _P(lhs) - _P(rhs)
    return _P(s)


def _sym(name: Any) -> sp.Symbol:
    name = str(name).strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(f"bad variable name '{name}'")
    return sp.Symbol(name)


def _fmt(value: Any) -> str:
    out = sp.sstr(value)
    try:
        if getattr(value, "is_number", False) and not value.is_Integer and not value.is_Rational:
            out += f"  (≈ {sp.N(value, 12)})"
    except Exception:
        pass
    return out

def t_calc(expression: str) -> str:
    return _fmt(sp.simplify(_P(expression)))


def t_solve_equations(equations: list[str], variables: list[str], domain: str = "complex") -> str:
    eqs = [_eq(e) for e in equations]
    syms = [_sym(v) for v in variables]
    if domain == "real" and len(eqs) == 1 and len(syms) == 1:
        return sp.sstr(sp.solveset(eqs[0], syms[0], domain=sp.S.Reals))
    return sp.sstr(sp.solve(eqs, syms, dict=True))

def t_wolfram(query: str) -> str:
    app_id = os.getenv("WOLFRAM_APP_ID")
    if not app_id:
        return "Agent API key is not configured."
    try:
        client = wolframalpha.Client(app_id)
        res = client.query(query)

        # 1. Try to get the primary result
        if res.results:
            try:
                answer = next(res.results).text
                if answer:
                    return f"Agent Answer: {answer}"
            except StopIteration:
                pass

        # 2. If primary result is empty, look for any text in the pods
        for pod in res.pods:
            if pod.text:
                return f"Agent Answer: {pod.text}"

        return "Agent returned no text results for this query (it might only have a graph/image)."
    except Exception as e:
        return f"Agent could not solve this: {str(e)}"

def t_solve_inequality(inequality: str, variable: str) -> str:
    rel = _P(inequality)
    v = _sym(variable)
    try:
        return sp.sstr(sp.solve_univariate_inequality(rel, v, relational=False))
    except Exception:
        return sp.sstr(sp.reduce_inequalities(rel, v))


def t_differentiate(expression: str, variable: str, order: int = 1) -> str:
    return _fmt(sp.simplify(sp.diff(_P(expression), _sym(variable), int(order))))


def t_integrate(expression: str, variable: str, lower: str | None = None, upper: str | None = None) -> str:
    f, v = _P(expression), _sym(variable)
    if lower is None or upper is None:
        return sp.sstr(sp.simplify(sp.integrate(f, v))) + "  (+ C)"
    lo, hi = _P(lower), _P(upper)
    res = sp.integrate(f, (v, lo, hi))
    if res.has(sp.Integral):
        return f"no closed form found; numeric ≈ {sp.N(sp.Integral(f, (v, lo, hi)), 12)}"
    return _fmt(sp.simplify(res))


def t_limit(expression: str, variable: str, point: str, direction: str = "+-") -> str:
    return _fmt(sp.limit(_P(expression), _sym(variable), _P(point), direction))


def t_series(expression: str, variable: str, point: str = "0", order: int = 6) -> str:
    return sp.sstr(sp.series(_P(expression), _sym(variable), _P(point), int(order)))


def t_sum_product(expression: str, variable: str, lower: str, upper: str, kind: str = "sum") -> str:
    f, v, lo, hi = _P(expression), _sym(variable), _P(lower), _P(upper)
    res = sp.product(f, (v, lo, hi)) if kind == "product" else sp.summation(f, (v, lo, hi))
    return _fmt(sp.simplify(res))


def t_substitute(expression: str, values: dict[str, str]) -> str:
    f = _P(expression)
    subs = {_sym(k): _P(v) for k, v in values.items()}
    return _fmt(sp.simplify(f.subs(subs)))


def t_check_equal(a: str, b: str) -> str:
    diff = sp.simplify(_P(a) - _P(b))
    if diff == 0:
        return "EQUAL (symbolic)"
    syms = sorted(diff.free_symbols, key=str)
    if not syms:
        return "EQUAL (numeric)" if abs(sp.N(diff, 15)) < 1e-9 else f"NOT EQUAL: difference = {_fmt(diff)}"
    rng, good, bad = random.Random(7), 0, None
    for _ in range(8):
        point = {s: sp.Rational(rng.randint(5, 40), 10) for s in syms}
        try:
            val = abs(sp.N(diff.subs(point), 15))
        except Exception:
            continue
        if val.is_number and val < 1e-9:
            good += 1
        elif val.is_number:
            bad = (point, val)
            break
    if bad is None and good >= 4:
        return f"EQUAL (numeric, {good} random points)"
    return f"NOT EQUAL: difference = {_fmt(diff)}"


def t_extremum(expression: str, variable: str, lower: str | None = None, upper: str | None = None) -> str:
    f, v = _P(expression), _sym(variable)
    interval = sp.Interval(_P(lower), _P(upper)) if lower is not None and upper is not None else sp.S.Reals
    crit = sp.solveset(sp.diff(f, v), v, interval)
    if not isinstance(crit, sp.FiniteSet):
        return f"critical points (not finite set): {sp.sstr(crit)}"
    pts = list(crit)
    if lower is not None and upper is not None:
        pts += [_P(lower), _P(upper)]
    rows = []
    for p in pts:
        val = sp.simplify(f.subs(v, p))
        d2 = sp.simplify(sp.diff(f, v, 2).subs(v, p))
        rows.append((p, val, d2))
    rows.sort(key=lambda r: float(sp.N(r[1])) if r[1].is_number else 0.0)
    lines = [f"f({sp.sstr(p)}) = {_fmt(val)}; f''={_fmt(d2)}" for p, val, d2 in rows]
    return "points checked (sorted by value, smallest first):\n" + "\n".join(lines)


def t_matrix_op(matrix: str, op: str) -> str:
    m = sp.Matrix(_P(matrix))
    ops: dict[str, Callable[[], Any]] = {
        "det": m.det, "inverse": m.inv, "rank": m.rank, "eigenvals": m.eigenvals,
        "eigenvects": m.eigenvects, "transpose": m.transpose, "trace": m.trace,
        "rref": lambda: m.rref()[0], "nullspace": m.nullspace,
        "charpoly": lambda: m.charpoly().as_expr(),
    }
    if op not in ops:
        raise ValueError(f"op must be one of {sorted(ops)}")
    return sp.sstr(ops[op]())


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}
REGISTRY: dict[str, tuple[Callable[..., str], str, dict]] = {
    "calc": (t_calc, "Simplify/evaluate an expression exactly (also gives a decimal). Use for ALL arithmetic, "
                     "binomial(n,r), factorial(n), complex numbers (use I), Matrix(...) expressions.",
             _schema({"expression": _S}, ["expression"])),
    "solve_equations": (t_solve_equations, "Solve equation(s) like 'x**2 - 5*x + 6 = 0'. domain: 'real' or 'complex'. "
                                           "Use real for one equation in one variable when roots must be real.",
                        _schema({"equations": {"type": "array", "items": _S}, "variables": {"type": "array", "items": _S},
                                 "domain": {"type": "string", "enum": ["real", "complex"]}}, ["equations", "variables"])),
    "solve_inequality": (t_solve_inequality, "Solve a one-variable inequality, e.g. 'x**2 - 5*x + 6 < 0'.",
                         _schema({"inequality": _S, "variable": _S}, ["inequality", "variable"])),
    "differentiate": (t_differentiate, "Derivative of expression w.r.t. variable (optional order).",
                      _schema({"expression": _S, "variable": _S, "order": {"type": "integer"}}, ["expression", "variable"])),
    "integrate": (t_integrate, "Indefinite integral, or definite if lower and upper are given (use 'oo' for infinity).",
                  _schema({"expression": _S, "variable": _S, "lower": _S, "upper": _S}, ["expression", "variable"])),
    "limit": (t_limit, "Limit of expression as variable -> point. direction: '+', '-' or '+-'.",
              _schema({"expression": _S, "variable": _S, "point": _S, "direction": _S}, ["expression", "variable", "point"])),
    "series": (t_series, "Taylor/Laurent series of expression around point up to the given order.",
               _schema({"expression": _S, "variable": _S, "point": _S, "order": {"type": "integer"}}, ["expression", "variable"])),
    "sum_product": (t_sum_product, "Finite or infinite sum/product of expression over variable from lower to upper.",
                    _schema({"expression": _S, "variable": _S, "lower": _S, "upper": _S,
                             "kind": {"type": "string", "enum": ["sum", "product"]}}, ["expression", "variable", "lower", "upper"])),
    "substitute": (t_substitute, "Substitute values into an expression, e.g. values {'x': '2', 'y': '1/2'}. "
                                 "Use to plug an answer back into the original conditions.",
                   _schema({"expression": _S, "values": {"type": "object", "additionalProperties": _S}}, ["expression", "values"])),
    "check_equal": (t_check_equal, "Check whether two expressions are equal (symbolic, then numeric). "
                                   "Use this to VERIFY a final answer against an independent computation.",
                    _schema({"a": _S, "b": _S}, ["a", "b"])),
    "extremum": (t_extremum, "Critical points and values for max/min of f(variable), optionally on [lower, upper] "
                             "(endpoints are included).",
                 _schema({"expression": _S, "variable": _S, "lower": _S, "upper": _S}, ["expression", "variable"])),
    "matrix_op": (t_matrix_op, "Matrix operation on a matrix like '[[1,2],[3,4]]'. "
                               "op: det, inverse, rank, eigenvals, eigenvects, transpose, trace, rref, nullspace, charpoly.",
                  _schema({"matrix": _S, "op": _S}, ["matrix", "op"])),

    "advanced_math": (t_wolfram, "Query Agent for advanced math, integrals, PDEs, or when SymPy fails. "
                           "Use standard math syntax (e.g. 'integrate x^3/(e^x - 1) from 0 to infinity').",
                _schema({"query": _S}, ["query"])),

}
TOOL_SPECS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
    for n, (_, d, p) in REGISTRY.items()
]


def _child(conn, fn, kwargs) -> None:
    try:
        conn.send(("ok", fn(**kwargs)))
    except BaseException as error:
        conn.send(("err", f"{type(error).__name__}: {error}"))
    finally:
        conn.close()


def run_tool(name: str, kwargs: dict, timeout: float = TOOL_TIMEOUT) -> tuple[bool, str]:
    if name not in REGISTRY:
        return False, f"unknown tool '{name}'. Available: {', '.join(REGISTRY)}"
    fn = REGISTRY[name][0]
    try:
        ctx = multiprocessing.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child, args=(child, fn, kwargs), daemon=True)
        proc.start()
        child.close()
    except (ValueError, OSError, AssertionError):
        try:
            return True, _clip(fn(**kwargs))
        except Exception as error:
            return False, f"{type(error).__name__}: {error}"
    try:
        if not parent.poll(timeout):
            proc.terminate()
            return False, f"timed out after {timeout:.0f}s; try a simpler or more specific computation"
        status, value = parent.recv()
    except EOFError:
        return False, "tool crashed"
    finally:
        proc.join(1)
        parent.close()
    return status == "ok", _clip(value)


def _clip(text: str) -> str:
    text = str(text)
    return text if len(text) <= MAX_TOOL_OUTPUT else text[:MAX_TOOL_OUTPUT] + " …[truncated]"

SYSTEM_PROMPT = """You are MathMind's deep-solve engine for JEE Main/Advanced level mathematics.

Method:
1. Restate what is asked in one line. For multiple-choice, note the options.
2. Identify the concepts/theorems involved and choose a plan.
3. Do EVERY calculation with the tools (algebra, calculus, matrices, counting, numeric checks). Never do non-trivial arithmetic in your head. If SymPy tools fail, hang, or return an error, IMMEDIATELY use the `advanced_math` tool as a fallback.
4. VERIFY before answering: substitute your answer back into the original conditions (substitute), or recompute a different way and compare with check_equal. Check domain restrictions, extraneous roots, signs, and edge cases. If a check fails, fix the solution.
5. For multiple-choice, test each option rather than guessing.
6. SANITY-CHECK every derived formula against what the problem implies BEFORE using it: sign (an integral of a positive function over a positive range is positive; lengths, areas and probabilities cannot be negative; a probability is in [0,1]), domain, and size. If a derived formula contradicts such a fact, your setup or algebra is WRONG: find the mistake. Never "match", "approximate" or drop terms to force agreement with a given option or a desired value.
7. PROBABILITY TRAP WARNING: When a problem says "randomly distributed", each object is distributed independently, even if the problem says they are "identical". To find the correct probability, you MUST temporarily label the identical objects to find the true physical sample space (e.g., 3^4 for 4 balls in 3 boxes). Do NOT use Stars and Bars to calculate probability weights unless the problem explicitly states that all visually distinct configurations are equally likely. Identical objects in a random experiment follow Maxwell-Boltzmann statistics for probability (like distinct particles), not Bose-Einstein.
8. If you cannot reach a rigorous, verified answer, do not guess. Make the last line exactly: FINAL_ANSWER: UNSURE, and explain in the body what you could not establish.

Tool syntax: plain text only. Use * for multiplication, ** or ^ for powers, sqrt(), pi, I (imaginary unit), E (Euler's number), oo (infinity). No LaTeX in tool arguments, no attribute access (x.is_real), no quotes.

Final response format (Markdown):
- Short numbered solution steps with the key reasoning. Use $...$ for inline math and $$...$$ for display math. Never use \\( \\) or \\[ \\].
- Never put $$ display equations inside list items; keep list items short and put display equations in their own paragraph.
- Start headings on a new line.
- The LAST line must be exactly: FINAL_ANSWER: <the answer only, e.g. 3/4, (B), or x = 2 or x = 3>
- IMPORTANT: Do not use line breaks inside math equations. Keep all equations on a single line unless you are using a \begin{aligned} environment.
Always use underscores for subscripts (e.g., write `x_0`, NOT `x0`).
-IMPORTANT: Do not use conversational filler like "MathMind:" or "Assistant:". Start directly with the solution steps.
- Always write limits correctly using `\lim_{x_0 \to \infty} \frac{...}{...}`.
Be honest: if you could not verify something or the problem is ambiguous, say so instead of guessing."""
class AgentError(RuntimeError):
    pass


_LIMIT_RE = re.compile(r"Limit\s+(\d+),\s*Requested\s+(\d+)", re.I)
MIN_COMPLETION_TOKENS = 1500


def _shrink_for_limit(error_text: str, max_tokens: int) -> int | None:
    """Groq counts max_completion_tokens against the per-minute token limit. On a 413
    ("Limit 8000, Requested 10016") return a smaller max_tokens that fits, or None."""
    m = _LIMIT_RE.search(error_text or "")
    if not m:
        return None
    limit, requested = int(m.group(1)), int(m.group(2))
    smaller = max_tokens - (requested - limit) - 200
    return smaller if smaller >= MIN_COMPLETION_TOKENS else None


def _chat(messages: list[dict], model: str, temperature: float, use_tools: bool = True,
          effort: str | None = None, max_tokens: int = 8192) -> dict:
    payload: dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_completion_tokens": max_tokens,
    }
    if use_tools:
        payload["tools"] = TOOL_SPECS
        payload["tool_choice"] = "auto"
    effort = (effort or REASONING_EFFORT)
    if "gpt-oss" in model and effort in {"low", "medium", "high"}:
        payload["reasoning_effort"] = effort
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"}
    for attempt in range(3):
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=(5, 70))
        if response.status_code == 400 and "reasoning_effort" in payload and attempt == 0:
            payload.pop("reasoning_effort")
            continue
        if response.status_code == 413 and attempt < 2:
            smaller = _shrink_for_limit(response.text, payload["max_completion_tokens"])
            if smaller:
                payload["max_completion_tokens"] = smaller
                continue
            raise AgentError("This question is too large for your Groq plan's per-minute token limit. "
                             "Shorten it, or upgrade the Groq tier.")
        break
    if response.status_code == 429:
        raise AgentError("The AI service is rate-limited right now (Groq's per-minute token limit). "
                         "Please try again in a minute, or set JEE_SAMPLES=1.")
    if response.status_code == 400 and "tool_use_failed" in response.text:
        raise ValueError("tool_use_failed")
    if response.status_code >= 400:
        raise AgentError(f"AI service error {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]


def _context(history: list[dict] | None) -> str:
    parts = [f"{m.get('role', '')}: {str(m.get('content', ''))[-1200:]}" for m in (history or [])[-6:]]
    return "\n".join(parts)[-5000:]


VERIFIER_PROMPT = """You are an independent, ruthless, skeptical verifier of math answers. You receive a PROBLEM and a CLAIMED answer. Your job is to find FLAWS in the setup and reasoning. Do not trust the claim.

Rules:
1. Check with tools: plug the claimed answer back into the original conditions and/or solve the problem independently by a DIFFERENT method.
2. You MUST run at least one tool and base your verdict on its output.
3. UNIVERSAL TRAP CHECKLIST: Scan the problem and solution for these common traps:
   - PROBABILITY SAMPLE SPACE: If objects are "identical" but distributed "randomly", treating visually distinct arrangements as equally likely is a FATAL ERROR (Bose-Einstein vs Maxwell-Boltzmann). You must treat identical objects as distinct (e.g., using 3^N or Stirling numbers) unless stated otherwise.
   - SQUARE ROOT: Did they take a square root and forget the ±? Did they forget that sqrt(x^2) = |x|?
   - DOMAIN: Did they divide by zero or take the log/sqrt of a negative number without restricting the domain?
   - EXTRANEOUS ROOTS: If they squared both sides, did they plug the answers back in?
   - BOUNDARY CONDITIONS: For ODEs/PDEs, do they satisfy ALL initial/boundary conditions?
   - UNPROVEN ASSUMPTIONS: Did they assume a matrix is invertible, a series converges, or a function is continuous without proof?
4. Check completeness: every part asked for, every valid root or case, and for multiple choice, that the chosen option matches the computed value.
5. Tool syntax: plain text only. Use * for multiplication, ** or ^ for powers, sqrt(), pi, I, E, oo. No LaTeX.
6. Reply with ONLY a JSON object, no other text:
{"verdict": "correct" | "incorrect" | "inconclusive", "evidence": "<1-3 sentences citing tool results and the specific trap violated>", "corrected_answer": "<the right answer if you found one, else empty>"}
Use "inconclusive" if you could not decide. Never answer "correct" without tool evidence."""


def _agent_loop(system: str, user: str, temperature: float, deadline: float, max_steps: int, on_event: Callable | None = None) -> dict:
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    trace: list[str] = []
    verdicts: list[str] = []
    format_errors = 0

    for _ in range(max_steps + 1):
        if time.monotonic() > deadline:
            raise AgentError("This problem took too long to solve. Try breaking it into smaller parts.")
        try:
            message = _chat(messages, REASONING_MODEL, temperature)
        except ValueError:
            format_errors += 1
            if format_errors > 2:
                raise AgentError("The model produced malformed tool calls. Please try again.")
            messages.append({"role": "user", "content": "Your last tool call was malformed. "
                                                        "Call the tool again with valid JSON arguments."})
            continue
        except requests.RequestException as error:
            raise AgentError(f"Could not reach the AI service: {error}") from error

        calls = message.get("tool_calls") or []
        if not calls:
            content = (message.get("content") or "").strip()
            if not content:
                raise AgentError("The model returned an empty answer.")
            return {"content": content, "trace": trace, "verdicts": verdicts}

        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args: Any = {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")

                if on_event:
                    pretty_name = name.replace("_", " ").title()
                    on_event("tool", f"Running {pretty_name}...")

                ok, output = run_tool(name, args)
            except (ValueError, TypeError) as error:
                ok, output = False, f"bad arguments: {error}"
            if name == "check_equal" and ok:
                verdicts.append("NOT EQUAL" if output.startswith("NOT") else "EQUAL")
            trace.append(f"{name}({json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)[:200]}) -> "
                         f"{'ok' if ok else 'error'}: {output[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": output if ok else f"ERROR: {output}"})

    messages.append({"role": "user", "content": "Stop using tools. Give your final response now, "
                                                "in exactly the format requested."})
    payload = {"model": REASONING_MODEL, "messages": messages, "temperature": temperature,
               "max_completion_tokens": 4000}
    response = requests.post(GROQ_URL, json=payload, timeout=(5, 60),
                             headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
    if response.status_code >= 400:
        raise AgentError("The solver ran out of steps before finishing.")
    content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    return {"content": content, "trace": trace, "verdicts": verdicts}


def _run_attempt(question: str, history: list[dict] | None, temperature: float, deadline: float,
                 feedback: str | None = None, on_event: Callable | None = None) -> dict:
    user = question
    ctx = _context(history)
    if ctx:
        user = f"Earlier conversation (for reference only):\n{ctx}\n\nProblem to solve now:\n{question}"
    if feedback:
        user += ("\n\nAn independent verifier checked your previous answer and found a problem:\n"
                 f"{feedback}\nRe-solve carefully. Do not simply accept the verifier's suggestion; "
                 "confirm any new answer with tools.")
    return _agent_loop(SYSTEM_PROMPT, user, temperature, deadline, MAX_STEPS, on_event=on_event)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def verify_solution(question: str, claimed: str, deadline: float | None = None) -> dict:
    deadline = deadline or time.monotonic() + 60
    user = f"PROBLEM:\n{question}\n\nCLAIMED ANSWER:\n{claimed}"
    try:
        run = _agent_loop(VERIFIER_PROMPT, user, 0.0, deadline, 6)
    except AgentError as error:
        return {"verdict": "unavailable", "evidence": str(error), "corrected": "", "tool_calls": 0}
    except Exception as error:
        logger.exception("verifier crashed")
        return {"verdict": "unavailable", "evidence": f"{type(error).__name__}", "corrected": "", "tool_calls": 0}
    ok_calls = sum(1 for line in run["trace"] if "-> ok:" in line)
    data = _parse_json(run["content"])
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in {"correct", "incorrect", "inconclusive"}:
        verdict = "inconclusive"
    evidence = str(data.get("evidence", "")).strip()[:400]
    if verdict == "correct" and ok_calls == 0:
        verdict, evidence = "inconclusive", "The verifier did not run any computation."
    return {"verdict": verdict, "evidence": evidence,
            "corrected": str(data.get("corrected_answer", "") or "").strip()[:200], "tool_calls": ok_calls}


AUDIT_PROMPT = """You are a math auditor. You receive a PROBLEM and a SOLUTION. Do NOT solve the problem and do NOT judge the final answer. Extract the checkable claims the SOLUTION relies on, so that a computer algebra system can test them.

Reply with ONLY a JSON object:
{"claims": [
  {"kind": "equal", "a": "<expr>", "b": "<expr>", "step": "<short description>"},
  {"kind": "sign", "expr": "<expr>", "relation": ">0", "domain": {"x": [0.1, 10]}, "step": "<why this must hold>"}
]}

Rules:
1. "equal": an algebraic identity the solution asserts, written with plain syntax (* for multiplication, ** or ^ for powers, sqrt(), pi, no LaTeX, no words). It must be unconditionally true for all values of its variables, or fully numeric. If an equality only holds under a condition, substitute the condition first or skip it. Skip approximations, asymptotics, inequalities and anything involving integrals/limits you cannot write as a closed expression.
2. "sign": a quantity whose sign or range follows from the PROBLEM (an integral of a positive function is positive, a length or probability is non-negative, a value that must be real and positive), written as the formula the SOLUTION derived for that quantity. relation is one of ">0", ">=0", "<0", "<=0", "!=0". "domain" gives numeric [low, high] for each variable, taken from the problem's constraints (e.g. x>0 -> [0.1, 10]).
3. Include a "sign" claim whenever the solution derives a closed-form for a quantity that the problem forces to be positive or negative. These catch contradictions.
4. At most 12 claims, most important first. If nothing is checkable, return {"claims": []}."""


def _extract_claims(question: str, body: str, deadline: float | None = None) -> list:
    user = f"PROBLEM:\n{question[:3000]}\n\nSOLUTION:\n{body[:6000]}"
    try:
        message = _chat([{"role": "system", "content": AUDIT_PROMPT}, {"role": "user", "content": user}],
                        REASONING_MODEL, 0.0, use_tools=False, effort="low", max_tokens=3000)
    except (AgentError, ValueError, requests.RequestException):
        return []
    claims = _parse_json(message.get("content") or "").get("claims")
    return claims if isinstance(claims, list) else []


def _tool_check_equal(a: str, b: str) -> str:
    ok, output = run_tool("check_equal", {"a": a, "b": b})
    return output if ok else f"ERROR: {output}"


def audit_solution(question: str, body: str, answer: str | None, deadline: float | None = None) -> dict:
    """Audit the solution's reasoning, not just its final answer.

    Returns {"status": "passed" | "failed" | "skipped", "checked": int, "failures": [str]}.
    "passed" means no contradiction was found among the claims actually checked; it is
    evidence, not a proof. Extraction/parse problems are skipped, never counted as failures.
    """
    failures = auditor.check_option_consistency(question, answer, body, _P)
    checked = 0
    if deadline is None or deadline - time.monotonic() > 12:
        claims = _extract_claims(question, body, deadline)
        result = auditor.run_claims(claims, _P, _tool_check_equal)
        failures += result["failures"]
        checked = result["checked"]
    if failures:
        return {"status": "failed", "checked": checked, "failures": failures}
    return {"status": "passed" if checked else "skipped", "checked": checked, "failures": []}


_FINAL = re.compile(r"^\s*\**FINAL_ANSWER\**\s*:\s*(.+?)\s*$", re.I | re.M)


def _split_answer(content: str) -> tuple[str, str | None]:
    matches = list(_FINAL.finditer(content))
    if not matches:
        return content.strip(), None
    last = matches[-1]
    body = (content[:last.start()] + content[last.end():]).strip()
    return body, last.group(1).strip().strip("`")


def _norm(answer: str) -> str:
    return re.sub(r"[\s$`*]|\\(?:left|right|displaystyle|boxed)", "", answer.lower())


def _is_unsure(answer: str | None) -> bool:
    return bool(answer) and answer.strip().strip("*` ").upper().startswith("UNSURE")


def _same(a: str, b: str) -> bool:
    if _norm(a) == _norm(b):
        return True
    try:
        return sp.simplify(_P(a) - _P(b)) == 0
    except Exception:
        return False


def _format_answer(answer: str) -> str:
    if "$" not in answer and re.search(r"\\|[_^]|frac", answer):
        return f"${answer}$"
    return answer


def _pick(runs: list[dict]) -> tuple[dict, tuple[int, int] | None]:
    answered = [r for r in runs if r["answer"] and not _is_unsure(r["answer"])]
    if len(answered) < 2:
        return (answered[0] if answered else runs[0]), None
    best = max(answered, key=lambda r: sum(_same(r["answer"], o["answer"]) for o in answered))
    return best, (sum(_same(best["answer"], o["answer"]) for o in answered), len(answered))


def solve_hard(question: str, history: list[dict] | None = None,
               normalize: Callable[[str], str] | None = None,
               on_event: Callable[[str, str], None] | None = None) -> dict:
    normalize = normalize or (lambda s: s)
    deadline = time.monotonic() + DEADLINE
    chosen: dict | None = None
    agreement: tuple[int, int] | None = None
    verdict: dict | None = None
    audit: dict | None = None
    error: AgentError | None = None
    feedback: str | None = None

    for round_no in range(MAX_FIX_ROUNDS + 1):
        runs: list[dict] = []
        try:
            for i in range(SAMPLES if round_no == 0 else 1):
                if on_event:
                    on_event("status", f"Solving (attempt {i+1})...")

                run = _run_attempt(question, history, 0.2 if i == 0 else 0.7, deadline, feedback, on_event=on_event)
                run["body"], run["answer"] = _split_answer(run["content"])
                runs.append(run)
                if i == 1 and runs[0]["answer"] and run["answer"] and _same(runs[0]["answer"], run["answer"]):
                    break
        except AgentError as exc:
            error = exc
        if not runs:
            break
        chosen, agreement = _pick(runs)

        if not VERIFY_ENABLED:
            verdict = None
            break
        if deadline - time.monotonic() < 15:
            verdict = {"verdict": "unavailable", "evidence": "ran out of time", "corrected": ""}
            break
        if _is_unsure(chosen["answer"]):
            verdict = {"verdict": "inconclusive", "evidence": "The solver reported it could not establish an answer.",
                       "corrected": ""}
            audit = None
            break

        if on_event:
            on_event("status", "Independently verifying the answer...")
        verdict = verify_solution(question, chosen["answer"] or chosen["body"][-1200:], deadline)

        audit = None
        if AUDIT_ENABLED:
            if on_event:
                on_event("status", "Auditing each solution step...")
            audit = audit_solution(question, chosen["body"], chosen["answer"], deadline)

        problems: list[str] = []
        if verdict["verdict"] == "incorrect":
            problems.append(verdict["evidence"] + (f" The verifier suggests: {verdict['corrected']}."
                                                   if verdict["corrected"] else ""))
        if audit and audit["status"] == "failed":
            problems.extend(f"Step audit: {f}" for f in audit["failures"][:4])
        if not problems:
            break
        feedback = "\n".join(problems)

    if chosen is None:
        message = str(error or "The solver failed.")
        return {"success": False, "operation": "deep_solve", "result": None, "steps": [],
                "verification": None, "response": message, "error": message}

    state = verdict["verdict"] if verdict else "off"
    evidence = (verdict or {}).get("evidence", "")
    audit_failed = bool(audit and audit["status"] == "failed")
    unsure = _is_unsure(chosen["answer"])

    if unsure:
        verification, note = False, "⚠️ The solver could not establish a rigorous answer, so none is given. Rephrase the problem or split it into parts."
    elif state == "correct" and not audit_failed:
        verification = True
        detail = f"; {audit['checked']} solution step(s) audited" if audit and audit["checked"] else ""
        note = "Verification: PASSED (independently checked with SymPy" + detail + ")" + (f"\n\n*Check:* {evidence}" if evidence else "")
    elif state == "incorrect":
        verification = False
        note = f"⚠️ Verification FAILED: {evidence} This answer is probably wrong, so do not rely on it."
    elif state == "correct" and audit_failed:
        verification = None
        note = "⚠️ The final answer passed an independent check, but the step audit found problems in the reasoning, so treat this as unconfirmed."
    elif state == "inconclusive":
        verification, note = None, "⚠️ Could not be independently verified. Double-check this answer."
    elif state == "unavailable":
        verification, note = None, f"⚠️ Verification was unavailable ({evidence}). Double-check this answer."
    else:
        verification, note = None, "⚠️ Verification is switched off (JEE_VERIFY=0). Double-check this answer."
    if audit_failed and not unsure:
        note += "\n\n**Step audit flagged:**\n" + "\n".join(f"- {f}" for f in audit["failures"][:4])
    if agreement and agreement[0] <= agreement[1] / 2:
        note += " Independent attempts also disagreed with each other."

    parts = [normalize(chosen["body"])]
    if chosen["answer"] and not unsure:
        label = "Final answer" if verification is True else "Unconfirmed answer"
        parts.append(f"**{label}:** {_format_answer(chosen['answer'])}")
    parts.append(note)
    response = "\n\n".join(p for p in parts if p)

    return {"success": True, "operation": "deep_solve",
            "result": None if unsure else chosen["answer"],
            "steps": chosen["trace"], "verification": verification, "verification_detail": verdict,
            "audit": audit, "response": response}
