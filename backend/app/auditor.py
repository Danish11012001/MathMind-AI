"""Deterministic audit helpers for MathMind.

Nothing in this module calls an LLM. The LLM only *extracts* claims from a
solution (see reasoning_agent.audit_solution); the pass/fail decision is made
here with SymPy, so a confident-sounding but wrong solution cannot talk its
way past the audit.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Callable

import sympy as sp

# --------------------------------------------------------------------------
# LaTeX -> plain SymPy-friendly text
# --------------------------------------------------------------------------

def _group(s: str, i: int) -> tuple[str, int] | None:
    """s[i] must be '{'. Return (content, index after the matching '}')."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return None


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] == " ":
        i += 1
    return i


def _brace_bare_args(s: str) -> str:
    """\\frac12 -> \\frac{1}{2}, \\sqrt3 -> \\sqrt{3} (single-token arguments without braces)."""
    s = re.sub(r"\\((?:d|t)?frac)\s*([0-9A-Za-z])\s*([0-9A-Za-z])", r"\\\1{\2}{\3}", s)
    s = re.sub(r"\\sqrt\s*([0-9A-Za-z])", r"\\sqrt{\1}", s)
    return s


def _expand_frac_sqrt(s: str) -> str:
    s = _brace_bare_args(s)
    for _ in range(40):  # nested structures: repeat until stable
        changed = False
        m = re.search(r"\\(?:d|t)?frac", s)
        if m:
            i = _skip_ws(s, m.end())
            a = _group(s, i)
            b = _group(s, _skip_ws(s, a[1])) if a else None
            if a and b:
                s = f"{s[:m.start()]}(({a[0]})/({b[0]})){s[b[1]:]}"
                changed = True
        m = re.search(r"\\sqrt", s)
        if m:
            i = m.end()
            root = None
            if i < len(s) and s[i] == "[":
                close = s.find("]", i)
                if close != -1:
                    root, i = s[i + 1:close], close + 1
            a = _group(s, _skip_ws(s, i))
            if a:
                rep = f"sqrt({a[0]})" if root is None else f"root({a[0]},{root})"
                s = f"{s[:m.start()]}{rep}{s[a[1]:]}"
                changed = True
        if not changed:
            break
    return s


_TEXT_SUBS = [
    (r"\\left|\\right|\\displaystyle|\\Bigl|\\Bigr|\\bigl|\\bigr|\\Big|\\big", ""),
    (r"\\[,;:! ]|\\quad|\\qquad", " "),
    (r"\\cdot|\\times", "*"),
    (r"\\div", "/"),
    (r"\\pi", "pi"),
    (r"\\infty", "oo"),
    (r"\\(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|ln|log|exp)", r"\1"),
    (r"\\arcsin", "asin"), (r"\\arccos", "acos"), (r"\\arctan", "atan"),
    (r"\\le(?:q)?\b", "<="), (r"\\ge(?:q)?\b", ">="), (r"\\neq?\b", "!="),
]


def latex_to_plain(text: str) -> str:
    """Best-effort conversion of simple LaTeX into SymPy-parsable text."""
    s = str(text).replace("$", "").replace("−", "-").replace("‑", "-")
    s = _expand_frac_sqrt(s)
    for pattern, repl in _TEXT_SUBS:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: "_" + re.sub(r"\W", "", m.group(1)), s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" .,;")

# --------------------------------------------------------------------------
# Multiple-choice consistency
# --------------------------------------------------------------------------

_OPT_MARK = re.compile(r"\(\s*([A-Da-d])\s*\)")


def extract_options(question: str) -> dict[str, str]:
    """Return {'A': '...', 'B': '...'} for questions written with (A) (B) (C) ... options."""
    marks = [(m.start(), m.end(), m.group(1).upper()) for m in _OPT_MARK.finditer(question)]
    best: list[tuple[int, int, str]] = []
    for start in range(len(marks)):
        if marks[start][2] != "A":
            continue
        run = [marks[start]]
        for mark in marks[start + 1:]:
            if mark[2] == chr(ord(run[-1][2]) + 1):
                run.append(mark)
        if len(run) >= 2:
            best = run  # prefer the last complete run
    options: dict[str, str] = {}
    for i, (_, end, letter) in enumerate(best):
        stop = best[i + 1][0] if i + 1 < len(best) else len(question)
        options[letter] = question[end:stop].strip().strip("$").strip().rstrip(".;,")
    return options


_KNOWN_NAMES = re.compile(r"(?<![A-Za-z])(?:sqrt|root|sin|cos|tan|cot|sec|csc|asin|acos|atan|sinh|cosh|tanh|"
                          r"exp|log|ln|pi|oo|abs|Abs|factorial|binomial|Rational)(?![A-Za-z])")


def _euler(text: str) -> str:
    """In a final answer a bare 'e' is Euler's number (SymPy spells it E)."""
    text = re.sub(r"(\d)e(?=\s*(?:\^|\*\*))", r"\1*E", text)  # 4e^2 -> 4*E^2 (1e5 is untouched)
    return re.sub(r"(?<![A-Za-z_0-9.])e(?![A-Za-z_0-9])", "E", text)


def values_equal(a: str, b: str, parse: Callable[[str], Any]) -> bool | None:
    """True/False when decidable, None when either side can't be parsed."""
    try:
        plain_a, plain_b = (_euler(latex_to_plain(a)), _euler(latex_to_plain(b)))
        # Prose (e.g. "the circle") is not a value; refuse to compare instead of guessing.
        if any(re.search(r"[A-Za-z]{3,}", _KNOWN_NAMES.sub("", t)) for t in (plain_a, plain_b)):
            return None
        pa, pb = parse(plain_a), parse(plain_b)
        diff = sp.simplify(pa - pb)
        if diff == 0:
            return True
        if not diff.free_symbols:
            return bool(abs(sp.N(diff, 15)) < 1e-9)
        return False
    except Exception:
        return None


_LETTER_ANSWER = re.compile(r"^\s*(?:option\s*)?\(?\s*([A-Da-d])\s*\)?\s*\.?\s*$", re.I)
_BODY_OPTION = re.compile(r"(?:option|choice|answer is|correct (?:option|answer) is)\s*\*{0,2}\s*\(?\s*([A-D])\s*\)?", re.I)


def check_option_consistency(question: str, answer: str | None, body: str,
                             parse: Callable[[str], Any]) -> list[str]:
    options = extract_options(question)
    if not options or not answer:
        return []
    issues: list[str] = []
    cleaned = answer.replace("$", "").replace("*", "").strip()
    m = _LETTER_ANSWER.match(cleaned)
    chosen = m.group(1).upper() if m else None
    if chosen is None:
        results = {k: values_equal(cleaned, v, parse) for k, v in options.items()}
        matches = [k for k, r in results.items() if r is True]
        if len(matches) == 1:
            chosen = matches[0]
        elif len(matches) > 1:
            return []
        elif all(r is False for r in results.values()):
            issues.append(f"The final answer '{answer}' does not equal any of the given options "
                          f"({', '.join(options)}); the setup or arithmetic is probably wrong.")
            return issues
    if chosen and chosen not in options:
        issues.append(f"The final answer names option ({chosen}), which does not exist.")
    mentioned = [x.upper() for x in _BODY_OPTION.findall(body or "")]
    if chosen and mentioned and mentioned[-1] != chosen:
        issues.append(f"The solution text points to option ({mentioned[-1]}) but the final answer is ({chosen}).")
    return issues

# --------------------------------------------------------------------------
# Sign / range sanity check
# --------------------------------------------------------------------------

_RELATIONS = {">0", ">=0", "<0", "<=0", "!=0"}


def check_sign(expr: Any, relation: str, domain: dict[str, tuple[float, float]] | None = None,
               samples: int = 40, seed: int = 11) -> tuple[str, str]:
    """Sample `expr` over the domain. Returns (status, detail); status is
    'holds', 'violated' or 'undecided'."""
    relation = relation.replace(" ", "")
    if relation not in _RELATIONS:
        return "undecided", f"unsupported relation '{relation}'"
    symbols = sorted(expr.free_symbols, key=str)
    domain = domain or {}
    ranges = {}
    for s in symbols:
        lo, hi = domain.get(str(s), (0.5, 5.0))
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
            lo, hi = 0.5, 5.0
        ranges[s] = (lo, hi)
    rng = random.Random(seed)
    valid = 0
    for _ in range(samples):
        point = {s: sp.Float(rng.uniform(lo, hi)) for s, (lo, hi) in ranges.items()}
        try:
            value = complex(sp.N(expr.subs(point), 15))
        except Exception:
            continue
        if not math.isfinite(value.real) or abs(value.imag) > 1e-9:
            continue
        v = value.real
        valid += 1
        ok = {">0": v > 1e-12, ">=0": v >= -1e-9, "<0": v < -1e-12,
              "<=0": v <= 1e-9, "!=0": abs(v) > 1e-12}[relation]
        if not ok:
            at = ", ".join(f"{k}={float(val):.4g}" for k, val in point.items()) or "constant"
            return "violated", f"value {v:.4g} at {at}"
    if valid < 8:
        return "undecided", "too few valid sample points"
    return "holds", f"{valid} sample points"

# --------------------------------------------------------------------------
# Run LLM-extracted claims deterministically
# --------------------------------------------------------------------------

MAX_CLAIMS = 12


def _domain_from(raw: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if isinstance(raw, dict):
        for name, bounds in raw.items():
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
                out[str(name)] = (lo, hi)
            except (TypeError, ValueError, IndexError):
                continue
    return out


def run_claims(claims: Any, parse: Callable[[str], Any],
               check_equal: Callable[[str, str], str]) -> dict:
    """Decide pass/fail for extracted claims. Parse failures are *skipped*, never
    counted as failures, so extraction mistakes cannot reject a correct solution."""
    failures: list[str] = []
    checked = skipped = 0
    if not isinstance(claims, list):
        return {"status": "skipped", "checked": 0, "skipped": 0, "failures": []}
    for claim in claims[:MAX_CLAIMS]:
        if not isinstance(claim, dict):
            skipped += 1
            continue
        step = str(claim.get("step", "")).strip()[:160]
        kind = str(claim.get("kind", "")).strip().lower()
        try:
            if kind == "equal":
                a, b = latex_to_plain(claim["a"]), latex_to_plain(claim["b"])
                out = check_equal(a, b)
                if out.startswith("NOT EQUAL"):
                    failures.append(f"Claimed '{a} = {b}' is false ({out[:120]}). Step: {step}")
                    checked += 1
                elif out.startswith("EQUAL"):
                    checked += 1
                else:
                    skipped += 1
            elif kind == "sign":
                expr = parse(latex_to_plain(claim["expr"]))
                status, detail = check_sign(expr, str(claim.get("relation", "")),
                                            _domain_from(claim.get("domain")))
                if status == "violated":
                    failures.append(f"'{latex_to_plain(claim['expr'])} {claim.get('relation')}' "
                                    f"is violated ({detail}). Step: {step}")
                    checked += 1
                elif status == "holds":
                    checked += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    status = "failed" if failures else ("passed" if checked else "skipped")
    return {"status": status, "checked": checked, "skipped": skipped, "failures": failures}
