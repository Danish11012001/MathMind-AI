"""Deep-solve agent for multi-step (JEE-style) problems.

The LLM reasons about the problem and calls exact SymPy tools for every
calculation. It must verify its answer with a tool before finishing. Tools are
parsed with a strict whitelist (no arbitrary Python) and run in a child process
with a timeout, so a runaway computation cannot hang the server.

Environment variables (all optional except GROQ_API_KEY):
    GROQ_REASONING_MODEL   default "openai/gpt-oss-120b"
    GROQ_REASONING_EFFORT  low | medium | high (default "high"; gpt-oss models)
    JEE_SAMPLES            1-3 independent attempts; >1 enables majority voting
    JEE_MAX_STEPS          max tool-calling rounds per attempt (default 10)
    JEE_DEADLINE_SECONDS   total time budget per question (default 85)
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
import re
import time
from typing import Any, Callable

import requests
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "high").strip().lower()
MAX_STEPS = int(os.getenv("JEE_MAX_STEPS", "10"))
SAMPLES = max(1, min(int(os.getenv("JEE_SAMPLES", "1")), 3))
DEADLINE = float(os.getenv("JEE_DEADLINE_SECONDS", "85"))
TOOL_TIMEOUT = 12
MAX_TOOL_OUTPUT = 1500


# --------------------------------------------------------------------------- #
# Availability and routing
# --------------------------------------------------------------------------- #

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
    """True for something that reads like a math problem (not chit-chat or a follow-up)."""
    return bool(_MATH_CUE.search(text)) and not _is_followup(text, history)


def is_hard_problem(text: str, history: list[dict] | None = None) -> bool:
    """Long, word-problem-style input that the single-operation planner can't handle."""
    words = len(text.split())
    return (len(text) >= 110 or words >= 18) and looks_like_problem(text, history)


# --------------------------------------------------------------------------- #
# Safe parsing
# --------------------------------------------------------------------------- #

_FUNCS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec,
    "csc": sp.csc, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "acot": sp.acot, "atan2": sp.atan2, "sinh": sp.sinh, "cosh": sp.cosh,
    "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "cbrt": sp.cbrt,
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
    """Parse a math expression from the model. Raises ValueError with a helpful hint."""
    s = str(text).strip()
    if not s:
        raise ValueError("empty expression")
    if len(s) > 800:
        raise ValueError("expression too long")
    s = (s.replace("−", "-").replace("×", "*").replace("÷", "/").replace("π", "pi")
          .replace("²", "**2").replace("³", "**3").replace("√", "sqrt"))
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
    except Exception as error:  # SyntaxError, TypeError, SympifyError, ...
        raise ValueError(f"could not parse '{s}': {error}") from error


def _eq(text: Any) -> Any:
    """'lhs = rhs' -> lhs - rhs; otherwise treat as an expression equal to 0."""
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


# --------------------------------------------------------------------------- #
# Tools (each returns a short string)
# --------------------------------------------------------------------------- #

def t_calc(expression: str) -> str:
    return _fmt(sp.simplify(_P(expression)))


def t_solve_equations(equations: list[str], variables: list[str], domain: str = "complex") -> str:
    eqs = [_eq(e) for e in equations]
    syms = [_sym(v) for v in variables]
    if domain == "real" and len(eqs) == 1 and len(syms) == 1:
        return sp.sstr(sp.solveset(eqs[0], syms[0], domain=sp.S.Reals))
    return sp.sstr(sp.solve(eqs, syms, dict=True))


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
    """Symbolic then numeric equality test. Use it to verify answers."""
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
}
TOOL_SPECS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
    for n, (_, d, p) in REGISTRY.items()
]


def _child(conn, fn, kwargs) -> None:
    try:
        conn.send(("ok", fn(**kwargs)))
    except BaseException as error:  # noqa: BLE001 - report everything to the model
        conn.send(("err", f"{type(error).__name__}: {error}"))
    finally:
        conn.close()


def run_tool(name: str, kwargs: dict, timeout: float = TOOL_TIMEOUT) -> tuple[bool, str]:
    """Run a tool in a forked child with a timeout. Returns (ok, text)."""
    if name not in REGISTRY:
        return False, f"unknown tool '{name}'. Available: {', '.join(REGISTRY)}"
    fn = REGISTRY[name][0]
    try:
        ctx = multiprocessing.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child, args=(child, fn, kwargs), daemon=True)
        proc.start()
        child.close()
    except (ValueError, OSError, AssertionError):  # no fork (e.g. Windows): run inline
        try:
            return True, _clip(fn(**kwargs))
        except Exception as error:  # noqa: BLE001
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


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are MathMind's deep-solve engine for JEE Main/Advanced level mathematics.

Method:
1. Restate what is asked in one line. For multiple-choice, note the options.
2. Identify the concepts/theorems involved and choose a plan.
3. Do EVERY calculation with the tools (algebra, calculus, matrices, counting, numeric checks). Never do non-trivial arithmetic in your head.
4. VERIFY before answering: substitute your answer back into the original conditions (substitute), or recompute a different way and compare with check_equal. Check domain restrictions, extraneous roots, signs, and edge cases. If a check fails, fix the solution.
5. For multiple-choice, test each option rather than guessing.

Tool syntax: plain text only. Use * for multiplication, ** or ^ for powers, sqrt(), pi, I (imaginary unit), E (Euler's number), oo (infinity). No LaTeX in tool arguments, no attribute access (x.is_real), no quotes.

Final response format (Markdown):
- Short numbered solution steps with the key reasoning. Use $...$ for inline math and $$...$$ for display math. Never use \\( \\) or \\[ \\].
- Never put $$ display equations inside list items; keep list items short and put display equations in their own paragraph.
- Start headings on a new line.
- The LAST line must be exactly: FINAL_ANSWER: <the answer only, e.g. 3/4, (B), or x = 2 or x = 3>
Be honest: if you could not verify something or the problem is ambiguous, say so instead of guessing."""


class AgentError(RuntimeError):
    pass


def _chat(messages: list[dict], model: str, temperature: float) -> dict:
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "tools": TOOL_SPECS, "tool_choice": "auto",
        "temperature": temperature, "max_completion_tokens": 6000,
    }
    if "gpt-oss" in model and REASONING_EFFORT in {"low", "medium", "high"}:
        payload["reasoning_effort"] = REASONING_EFFORT
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"}
    for attempt in range(2):
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=(5, 70))
        if response.status_code == 400 and "reasoning_effort" in payload and attempt == 0:
            payload.pop("reasoning_effort")  # model does not support it
            continue
        break
    if response.status_code == 429:
        raise AgentError("The AI service is rate-limited right now. Please try again in a minute.")
    if response.status_code == 400 and "tool_use_failed" in response.text:
        raise ValueError("tool_use_failed")
    if response.status_code >= 400:
        raise AgentError(f"AI service error {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]


def _context(history: list[dict] | None) -> str:
    parts = [f"{m.get('role', '')}: {str(m.get('content', ''))[-1200:]}" for m in (history or [])[-6:]]
    return "\n".join(parts)[-5000:]


def _run_attempt(question: str, history: list[dict] | None, temperature: float, deadline: float) -> dict:
    user = question
    ctx = _context(history)
    if ctx:
        user = f"Earlier conversation (for reference only):\n{ctx}\n\nProblem to solve now:\n{question}"
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    trace: list[str] = []
    verdicts: list[str] = []
    format_errors = 0

    for _ in range(MAX_STEPS + 1):
        if time.monotonic() > deadline:
            raise AgentError("This problem took too long to solve. Try breaking it into smaller parts.")
        try:
            message = _chat(messages, REASONING_MODEL, temperature)
        except ValueError:  # malformed tool call from the model
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
                ok, output = run_tool(name, args)
            except (ValueError, TypeError) as error:
                ok, output = False, f"bad arguments: {error}"
            if name == "check_equal" and ok:
                verdicts.append("NOT EQUAL" if output.startswith("NOT") else "EQUAL")
            trace.append(f"{name}({json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)[:200]}) -> "
                         f"{'ok' if ok else 'error'}: {output[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": output if ok else f"ERROR: {output}"})

    # Out of steps: ask for a final answer with no more tools.
    messages.append({"role": "user", "content": "Stop using tools. Write the final solution now, "
                     "ending with the FINAL_ANSWER line."})
    payload = {"model": REASONING_MODEL, "messages": messages, "temperature": temperature,
               "max_completion_tokens": 3000}
    response = requests.post(GROQ_URL, json=payload, timeout=(5, 60),
                             headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
    if response.status_code >= 400:
        raise AgentError("The solver ran out of steps before finishing.")
    content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    return {"content": content, "trace": trace, "verdicts": verdicts}


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


def solve_hard(question: str, history: list[dict] | None = None,
               normalize: Callable[[str], str] | None = None) -> dict:
    """Solve a multi-step problem. Returns the same dict shape as answer_math()."""
    normalize = normalize or (lambda s: s)
    deadline = time.monotonic() + DEADLINE
    runs: list[dict] = []
    error: AgentError | None = None

    for i in range(SAMPLES):
        try:
            run = _run_attempt(question, history, 0.2 if i == 0 else 0.7, deadline)
        except AgentError as exc:
            error = exc
            break
        run["body"], run["answer"] = _split_answer(run["content"])
        runs.append(run)
        if i == 1 and runs[0]["answer"] and run["answer"] and _same(runs[0]["answer"], run["answer"]):
            break  # two independent attempts agree; no third needed

    if not runs:
        message = str(error or "The solver failed.")
        return {"success": False, "operation": "deep_solve", "result": None, "steps": [],
                "verification": None, "response": message, "error": message}

    # Majority vote among attempts that produced an answer.
    chosen, agreement = runs[0], None
    answered = [r for r in runs if r["answer"]]
    if len(answered) > 1:
        best = max(answered, key=lambda r: sum(_same(r["answer"], o["answer"]) for o in answered))
        votes = sum(_same(best["answer"], o["answer"]) for o in answered)
        chosen, agreement = best, (votes, len(answered))

    verdicts = chosen["verdicts"]
    tool_verified = "EQUAL" in verdicts and "NOT EQUAL" not in verdicts[-1:]
    consistent = agreement is not None and agreement[0] == agreement[1]

    failed_check = bool(verdicts) and verdicts[-1] == "NOT EQUAL"

    if failed_check:
        verification = False
        note = "⚠️ A verification check failed and was not resolved. Treat this answer as uncertain."
    elif tool_verified and (agreement is None or consistent):
        verification, note = True, "Verification: PASSED (checked with SymPy)"
    elif agreement is not None and agreement[0] > agreement[1] / 2:
        verification = True
        note = f"Verification: PASSED ({agreement[0]} of {agreement[1]} independent attempts agree)"
    elif agreement is not None:
        verification = False
        note = (f"⚠️ Attempts disagreed ({agreement[0]} of {agreement[1]} matched). "
                "Treat this answer as uncertain and double-check it.")
    else:
        verification = None
        note = "⚠️ Not independently verified. Double-check important results."

    parts = [normalize(chosen["body"])]
    if chosen["answer"]:
        parts.append(f"**Final answer:** {_format_answer(chosen['answer'])}")
    parts.append(note)
    response = "\n\n".join(p for p in parts if p)

    return {"success": True, "operation": "deep_solve", "result": chosen["answer"],
            "steps": chosen["trace"], "verification": verification, "response": response}
