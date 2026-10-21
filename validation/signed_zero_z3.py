"""Signed-zero gap between the two Rust reflector formulas.

de Moura and Bjorner, Z3, TACAS 2008, doi:10.1007/978-3-540-78800-3_24.

(i) -(g - t * (2 * (g·t))) and (-g) - t * (2 * ((-g)·t)) on binary64.
(ii) x + 0.0, the zero DNEB add in climbing_image_force.

The queries are quantifier-free floating-point satisfiability of the
negated claim, so unsat means the claim holds for every binary64 input.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

from z3 import (
    And,
    BitVecVal,
    FP,
    FPSort,
    FPVal,
    Not,
    Or,
    RNE,
    Solver,
    fpAdd,
    fpIsInf,
    fpIsNaN,
    fpIsNegative,
    fpIsZero,
    fpMul,
    fpNeg,
    fpSub,
    fpToIEEEBV,
    sat,
    unsat,
)

SORT = FPSort(11, 53)
RM = RNE()
TWO = FPVal(2.0, SORT)
POS0 = FPVal(0.0, SORT)


def solve(claim, timeout_ms):
    solver = Solver()
    solver.set("timeout", timeout_ms)
    solver.add(Not(claim))
    result = solver.check()
    model = None
    if result == sat:
        model = solver.model()
    return result, model


def reflect_forms(g, t):
    """Scalar stand-in for one component when the dot product is g*t."""
    dot = fpMul(RM, g, t)
    scaled = fpMul(RM, TWO, dot)
    inner = fpSub(RM, g, fpMul(RM, t, scaled))
    form_a = fpNeg(inner)
    neg_g = fpNeg(g)
    dot_b = fpMul(RM, neg_g, t)
    scaled_b = fpMul(RM, TWO, dot_b)
    form_b = fpSub(RM, neg_g, fpMul(RM, t, scaled_b))
    return form_a, form_b


def bits_or_signed_zero(form_a, form_b):
    both_zero = And(fpIsZero(form_a), fpIsZero(form_b))
    same_bits = fpToIEEEBV(form_a) == fpToIEEEBV(form_b)
    return Or(same_bits, both_zero)


def status_name(result):
    if result == unsat:
        return "unsat"
    if result == sat:
        return "sat"
    return "unknown"


def main():
    g = FP("g", SORT)
    t = FP("t", SORT)
    form_a, form_b = reflect_forms(g, t)
    finite = And(Not(fpIsNaN(g)), Not(fpIsInf(g)), Not(fpIsNaN(t)), Not(fpIsInf(t)))
    # Full finite domain. Bit-blasting binary64 multiplication did not finish
    # in three minutes on this solver, so the timeout stays short and an
    # unknown result is recorded rather than treated as a proof.
    claim_i = Or(Not(finite), bits_or_signed_zero(form_a, form_b))
    status_i, model_i = solve(claim_i, 20000)
    print("reflect forms, finite binary64:", status_i)
    if status_i == sat:
        print("counterexample", model_i)

    both_zero = And(fpIsZero(g), fpIsZero(t))
    claim_zero = Or(Not(both_zero), bits_or_signed_zero(form_a, form_b))
    status_zero, model_zero = solve(claim_zero, 20000)
    print("reflect forms, both inputs zero:", status_zero)
    if status_zero == sat:
        print("counterexample", model_zero)
    if status_zero != unsat:
        raise SystemExit(f"signed-zero reflector query was not proved: {status_zero}")

    x = FP("x", SORT)
    added = fpAdd(RM, x, POS0)
    same = fpToIEEEBV(added) == fpToIEEEBV(x)
    neg_zero = And(fpIsZero(x), fpIsNegative(x))
    pos_zero = And(fpIsZero(added), Not(fpIsNegative(added)))
    claim_ii = Or(same, And(neg_zero, pos_zero))
    status_ii, model_ii = solve(claim_ii, 20000)
    print("x + 0.0:", status_ii)
    if status_ii == sat:
        print("counterexample", model_ii)

    # The zero-add claim is the one the climbing image needs. Refuse a
    # timeout or a counterexample. The general reflector claim is recorded
    # as proved only when that query is unsat.
    if status_ii != unsat:
        raise SystemExit(f"x + 0.0 was not proved: {status_ii}")
    proved_reflect = status_i == unsat
    if proved_reflect:
        reflect_statement = (
            "On finite binary64 scalars the two reflector formulas agree as IEEE bits, "
            "or both results are zero (the sign may differ)."
        )
    elif status_i == sat:
        reflect_statement = "Z3 found a finite counterexample; see the log."
    else:
        reflect_statement = (
            "Z3 returned unknown on the full finite domain within 20 seconds. "
            "The query restricted to signed-zero inputs is unsat: the two formulas "
            "agree as bits, or both results are zero."
        )

    out = {
        "reflect_forms_finite_binary64": status_name(status_i),
        "reflect_forms_signed_zero_inputs": "unsat",
        "reflect_forms_statement": reflect_statement,
        "add_pos_zero": "unsat",
        "add_pos_zero_statement": (
            "x + 0.0 keeps every binary64 bit pattern except -0, which becomes +0."
        ),
        "consequence": (
            "Adding a zero DNEB vector maps a -0 component to +0. "
            "On signed-zero scalar inputs the two reflector formulas agree up to that sign. "
            "Z3 did not decide the formulas on every finite binary64 value, "
            "so the Rust forms stay as written until a deciding query exists."
        ),
        "citation": "doi:10.1007/978-3-540-78800-3_24",
    }
    path = Path(__file__).resolve().parent / "golden" / "signed_zero_z3.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    # Touch an unused name so a missing import fails loudly in review.
    assert BitVecVal(0, 1) is not None


if __name__ == "__main__":
    main()
