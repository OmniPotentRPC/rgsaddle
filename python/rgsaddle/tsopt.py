# MIT License
# Copyright 2026 Rohit Goswami
"""Index-1 transition-state refinement with a counted Hessian.

Every call to ``energy_grad`` is one force evaluation. Every call to
``projected_hessian`` is one Hessian evaluation. A finite-difference
product inside the caller's Hessian is the caller's to count, on the
force side, before this refiner sees the matrix.

The free coordinates are the columns of the basis returned with the
projected Hessian. For a free cluster that basis is the rigid
quotient. For a slab with xy periodicity it is the complement of the
three translations.

``kind="default"`` is one restricted-step partitioned RFO on the exact
projected Hessian, rejecting a step that lands in a minimum or that
flattens a deep negative mode without cutting the gradient.
``kind="tuned"`` retries a rejected start with the ridge search, which
keeps a fraction of the negative curvature and, on a convex point,
maximizes the softest mode. ``kind="ridge"`` is that second search on
its own. The accepted point is index 1 when the lowest projected
eigenvalue is below ``-1e-6`` and the Cartesian gradient norm is below
the force tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORCE_TOL = 1e-3
NEG_CUT = -1e-6


class _Cap(RuntimeError):
    pass


@dataclass
class RefineResult:
    """One refinement. ``calls`` is force evaluations plus Hessian evaluations."""

    x: np.ndarray
    calls: int
    force_calls: int
    hessian_calls: int
    converged: bool
    index: int
    gradient_norm: float


def _maximize_component(curvature: float, component: float, alpha: float) -> float:
    disc = np.sqrt(curvature * curvature + 4.0 * alpha * component * component)
    mu = 0.5 * (curvature + disc)
    denom = mu - curvature
    if abs(component) < 1e-16 or abs(denom) < 1e-18:
        return 0.0
    return component / denom


def _minimize_components(evals: np.ndarray, grad: np.ndarray, alpha: float) -> np.ndarray:
    if evals.size == 0:
        return grad
    e0 = float(evals[0])
    g2 = float(grad @ grad)
    mu = 0.5 * (e0 - np.sqrt(e0 * e0 + 4.0 * alpha * g2))
    for _ in range(40):
        d = mu - evals
        if np.any(np.abs(d) < 1e-18):
            mu = e0 - 1e-8
            d = mu - evals
        f = -mu / alpha + float(np.sum(grad * grad / d))
        fp = -1.0 / alpha - float(np.sum(grad * grad / (d * d)))
        if abs(f) < 1e-12 * (1.0 + abs(mu)):
            break
        mu_new = mu - f / fp if fp != 0.0 else mu
        if not np.isfinite(mu_new) or mu_new >= e0:
            mu_new = 0.5 * (mu + e0) - 1e-12
        mu = mu_new
    return grad / (mu - evals)


def prfo_step(evals: np.ndarray, evecs: np.ndarray, gradient: np.ndarray, delta: float) -> np.ndarray:
    """Baker restricted-step partition. Alpha grows until the step fits ``delta``."""
    g_eig = evecs.T @ gradient

    def assemble(alpha: float) -> np.ndarray:
        coeff = np.empty_like(g_eig)
        coeff[0] = _maximize_component(float(evals[0]), float(g_eig[0]), alpha)
        if g_eig.size > 1:
            coeff[1:] = _minimize_components(evals[1:], g_eig[1:], alpha)
        step = evecs @ coeff
        if not np.all(np.isfinite(step)):
            return np.zeros_like(gradient)
        return step

    step = assemble(1.0)
    nrm = float(np.linalg.norm(step))
    if nrm <= delta or delta <= 0.0:
        return step
    lo, hi = 1.0, 2.0
    bracketed = False
    trial = step
    for _ in range(30):
        trial = assemble(hi)
        if float(np.linalg.norm(trial)) <= delta:
            bracketed = True
            break
        hi *= 2.0
    if not bracketed:
        return step * (delta / nrm) if nrm > 0.0 else step
    best = trial
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        trial = assemble(mid)
        if float(np.linalg.norm(trial)) > delta:
            lo = mid
        else:
            hi = mid
            best = trial
    return best


def rigid_basis(x: np.ndarray) -> np.ndarray:
    """Orthonormal complement of the three translations and three rotations."""
    n = x.size // 3
    pos = np.ascontiguousarray(x, dtype=np.float64).reshape(n, 3)
    com = pos.mean(axis=0)
    cols = []
    for axis in range(3):
        v = np.zeros(x.size)
        v[axis::3] = 1.0
        cols.append(v)
    rel = pos - com
    for axis in range(3):
        v = np.zeros(x.size)
        for i in range(n):
            v[3 * i : 3 * i + 3] = np.cross(rel[i], np.eye(3)[axis])
        cols.append(v)
    matrix = np.column_stack(cols)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular > 1e-8))
    return np.ascontiguousarray(left[:, rank:])


def translation_basis(ncoord: int) -> np.ndarray:
    """Orthonormal complement of the three uniform translations."""
    cols = []
    for axis in range(3):
        v = np.zeros(ncoord)
        v[axis::3] = 1.0
        cols.append(v)
    left, singular, _ = np.linalg.svd(np.column_stack(cols), full_matrices=True)
    rank = int(np.sum(singular > 1e-8))
    return np.ascontiguousarray(left[:, rank:])


class CallCounter:
    """Count force and Hessian requests around a caller's surface.

    ``projected_hessian`` must not call back into this counter. A
    finite-difference Hessian counts its own gradient samples on the
    raw callable, then one Hessian request here if the refiner asks
    for the matrix.
    """

    def __init__(self, energy_grad, projected_hessian):
        self._energy_grad = energy_grad
        self._projected_hessian = projected_hessian
        self.force_calls = 0
        self.hessian_calls = 0

    def energy_grad(self, x: np.ndarray):
        self.force_calls += 1
        return self._energy_grad(x)

    def projected_hessian(self, x: np.ndarray):
        self.hessian_calls += 1
        return self._projected_hessian(x)

    @property
    def calls(self) -> int:
        return self.force_calls + self.hessian_calls


def _finish(x, g, h, force: int, hess: int, tol: float) -> RefineResult:
    gn = float(np.linalg.norm(g))
    ev = np.linalg.eigvalsh(h)
    neg = int(np.sum(ev < NEG_CUT))
    return RefineResult(
        x=np.ascontiguousarray(x, dtype=np.float64).copy(),
        calls=force + hess,
        force_calls=force,
        hessian_calls=hess,
        converged=gn < tol and neg == 1,
        index=neg,
        gradient_norm=gn,
    )


def _failed(x, force: int, hess: int) -> RefineResult:
    return RefineResult(
        x=np.ascontiguousarray(x, dtype=np.float64).copy(),
        calls=force + hess,
        force_calls=force,
        hessian_calls=hess,
        converged=False,
        index=-1,
        gradient_norm=1e9,
    )


def refine_fast(
    x0: np.ndarray,
    energy_grad,
    projected_hessian,
    *,
    cap: int = 80,
    force_tol: float = FORCE_TOL,
) -> RefineResult:
    """Restricted-step P-RFO with an exact projected Hessian each accepted step."""
    force = 0
    hess = 0
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()

    def fg(z):
        nonlocal force
        if force + hess >= cap:
            raise _Cap
        force += 1
        return energy_grad(z)

    def hp(z):
        nonlocal hess
        if force + hess >= cap:
            raise _Cap
        hess += 1
        return projected_hessian(z)

    try:
        energy, g = fg(x)
        u, h = hp(x)
        delta = 0.18
        rejects = 0
        for _ in range(80):
            gn = float(np.linalg.norm(g))
            ev, vecs = np.linalg.eigh(h)
            neg = int(np.sum(ev < NEG_CUT))
            ev0 = float(ev[0])
            if gn < force_tol and neg == 1:
                break
            gf = u.T @ g
            limit = delta
            if neg == 1 and gn < 1.5:
                limit = min(limit, max(0.008, 0.28 * gn))
            if neg == 0:
                v = vecs[:, 0]
                sign = 1.0 if float(gf @ v) >= 0.0 else -1.0
                step = u @ (v * sign * min(0.08, max(limit, 0.02)))
            else:
                step = u @ prfo_step(ev, vecs, gf, limit)
            nrm = float(np.linalg.norm(step))
            if nrm < 1e-14:
                delta = max(delta * 0.5, 1e-4)
                rejects += 1
                if rejects > 8:
                    break
                continue
            pred = float(g @ step + 0.5 * step @ (u @ (h @ (u.T @ step))))
            e1, g1 = fg(x + step)
            g1n = float(np.linalg.norm(g1))
            rho = (e1 - energy) / pred if abs(pred) >= 1e-14 else 1.0
            rho_ok = 0.15 <= rho <= 6.0 or (neg == 0 and g1n < gn)
            if not rho_ok:
                delta = max(min(delta, nrm) * 0.5, 1e-4)
                rejects += 1
                if rejects > 8:
                    break
                continue
            u1, h1 = hp(x + step)
            ev1 = np.linalg.eigvalsh(h1)
            neg1 = int(np.sum(ev1 < NEG_CUT))
            ev1_0 = float(ev1[0])
            collapsed = neg >= 1 and ev0 < -0.25 and ev1_0 > 0.35 * ev0 and g1n > 0.75 * gn
            lost = neg1 == 0 and g1n >= force_tol
            if collapsed or lost:
                delta = max(min(delta, nrm) * 0.45, 1e-4)
                rejects += 1
                if rejects > 8:
                    break
                continue
            x = x + step
            energy, g, u, h = e1, g1, u1, h1
            rejects = 0
            if 0.65 < rho < 1.45 and nrm > 0.7 * limit and ev1_0 < min(-0.2, 0.85 * ev0):
                delta = min(max(delta, nrm) * 1.25, 0.4)
            elif rho < 0.45 or rho > 2.2:
                delta = max(min(delta, nrm) * 0.8, 1e-4)
        return _finish(x, g, h, force, hess, force_tol)
    except _Cap:
        return _failed(x, force, hess)


def refine_ridge(
    x0: np.ndarray,
    energy_grad,
    projected_hessian,
    *,
    cap: int = 160,
    force_tol: float = FORCE_TOL,
) -> RefineResult:
    """P-RFO that refuses a step which gives up a healthy negative mode."""
    force = 0
    hess = 0
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()

    def fg(z):
        nonlocal force
        if force + hess >= cap:
            raise _Cap
        force += 1
        return energy_grad(z)

    def hp(z):
        nonlocal hess
        if force + hess >= cap:
            raise _Cap
        hess += 1
        return projected_hessian(z)

    try:
        energy, g = fg(x)
        u, h = hp(x)
        delta = 0.18
        for _ in range(80):
            gn = float(np.linalg.norm(g))
            ev, vecs = np.linalg.eigh(h)
            neg = int(np.sum(ev < NEG_CUT))
            ev0 = float(ev[0])
            if gn < force_tol and neg == 1:
                break
            gf = u.T @ g
            limit = delta
            if neg == 1 and gn < 1.5:
                limit = min(limit, max(0.01, 0.3 * gn))
            if neg == 0:
                step = u @ prfo_step(ev, vecs, gf, max(limit, 0.08))
                v = vecs[:, 0]
                sign = 1.0 if float(gf @ v) >= 0.0 else -1.0
                pure = u @ (v * sign * min(0.12, max(limit, 0.04)))
            else:
                step = u @ prfo_step(ev, vecs, gf, limit)
                g0 = float(gf @ vecs[:, 0])
                pure_c = _maximize_component(ev0, g0, 1.0)
                pure = u @ (vecs[:, 0] * pure_c)
                pn = float(np.linalg.norm(pure))
                if pn > limit > 0.0:
                    pure *= limit / pn
            found = None
            for cand in (step, pure):
                for scale in (1.0, 0.5, 0.25, 0.125):
                    trial = cand * scale
                    nrm = float(np.linalg.norm(trial))
                    if nrm < 1e-10:
                        continue
                    pred = float(g @ trial + 0.5 * trial @ (u @ (h @ (u.T @ trial))))
                    e1, g1 = fg(x + trial)
                    g1n = float(np.linalg.norm(g1))
                    rho = (e1 - energy) / pred if abs(pred) >= 1e-14 else 1.0
                    rho_ok = 0.1 <= rho <= 8.0 or g1n < gn
                    if not rho_ok:
                        continue
                    u1, h1 = hp(x + trial)
                    ev1 = np.linalg.eigvalsh(h1)
                    neg1 = int(np.sum(ev1 < NEG_CUT))
                    ev1_0 = float(ev1[0])
                    if g1n < force_tol and neg1 == 1:
                        found = (trial, e1, g1, u1, h1, neg1, ev1_0, g1n, rho, nrm)
                        break
                    if neg1 < 1:
                        continue
                    if ev0 < -0.15 and not (ev1_0 < 0.6 * ev0):
                        continue
                    if ev0 >= -0.15 and g1n > gn * 0.995:
                        continue
                    found = (trial, e1, g1, u1, h1, neg1, ev1_0, g1n, rho, nrm)
                    break
                if found is not None:
                    break
            if found is None:
                delta = max(delta * 0.5, 1e-3)
                if delta <= 1e-3 and neg == 0:
                    break
                continue
            trial, energy, g, u, h, neg1, ev1_0, g1n, rho, nrm = found
            x = x + trial
            if 0.6 < rho < 1.5 and ev1_0 < min(-0.3, 0.9 * ev0):
                delta = min(max(delta, nrm) * 1.2, 0.35)
            elif rho < 0.4 or rho > 2.5:
                delta = max(min(delta, nrm), 1e-3)
        return _finish(x, g, h, force, hess, force_tol)
    except _Cap:
        return _failed(x, force, hess)


def refine(
    x0: np.ndarray,
    energy_grad,
    projected_hessian,
    *,
    kind: str = "tuned",
    cap: int = 80,
    fast_cap: int = 48,
    ridge_cap: int = 160,
    force_tol: float = FORCE_TOL,
) -> RefineResult:
    """Refine one Cartesian start to an index-1 saddle.

    ``kind`` is ``default`` (fast Newton), ``ridge`` (curvature-preserving
    Newton), or ``tuned`` (fast Newton, then the ridge search if that
    start is not index 1). Caps count force and Hessian requests together.
    """
    if kind == "default":
        return refine_fast(
            x0, energy_grad, projected_hessian, cap=cap, force_tol=force_tol
        )
    if kind == "ridge":
        return refine_ridge(
            x0, energy_grad, projected_hessian, cap=ridge_cap, force_tol=force_tol
        )
    if kind != "tuned":
        raise ValueError(f"unknown kind {kind}")
    first = refine_fast(
        x0, energy_grad, projected_hessian, cap=fast_cap, force_tol=force_tol
    )
    if first.converged:
        return first
    second = refine_ridge(
        x0, energy_grad, projected_hessian, cap=ridge_cap, force_tol=force_tol
    )
    return RefineResult(
        x=second.x,
        calls=first.calls + second.calls,
        force_calls=first.force_calls + second.force_calls,
        hessian_calls=first.hessian_calls + second.hessian_calls,
        converged=second.converged,
        index=second.index,
        gradient_norm=second.gradient_norm,
    )


def follow_soft_mode(
    x0: np.ndarray,
    mode: np.ndarray,
    energy_grad,
    projected_hessian,
    *,
    step: float = 0.08,
    perp_cap: float = 0.12,
    curvature_floor: float = 0.08,
    stop_curvature: float = -0.4,
    stop_parallel: float = 0.3,
    max_steps: int = 100,
    finish_steps: int = 12,
    finish_delta: float = 0.6,
) -> RefineResult:
    """Walk one free mode uphill, then finish with index-1 P-RFO.

    ``mode`` is a vector in the free basis returned with the projected
    Hessian, not a Cartesian displacement. A rising curvature on the
    first steps reverses the direction once. Perpendicular modes are
    Newton-minimized. A negative perpendicular curvature keeps its own
    eigenvalue so the step returns toward that mode's stationary point.
    The finisher takes a partitioned RFO inside ``finish_delta`` and
    keeps a step only when the free Hessian stays index 1.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    followed = np.ascontiguousarray(mode, dtype=np.float64).copy()
    followed /= max(float(np.linalg.norm(followed)), 1e-16)
    force = 0
    hess = 0
    anchor = None
    forced = None
    reversed_once = False
    reached = False
    last_h = None
    last_g = None
    it = 0
    while it < max_steps:
        _energy, g_i = energy_grad(x)
        force += 1
        basis, h_i = projected_hessian(x)
        hess += 1
        last_g = g_i
        last_h = h_i
        ev_i, vecs_i = np.linalg.eigh(h_i)
        gf = basis.T @ g_i
        overlaps = vecs_i.T @ followed
        k = int(np.argmax(np.abs(overlaps)))
        align = 1.0 if overlaps[k] >= 0.0 else -1.0
        followed = align * vecs_i[:, k]
        gpar = float(gf @ followed)
        curv = float(ev_i[k])
        if anchor is None:
            anchor = (x.copy(), followed.copy(), curv, 1.0 if gpar >= 0.0 else -1.0)
        if not reversed_once and it == 4 and curv > anchor[2] + 0.01:
            reversed_once = True
            x, followed, _curv0, first_dir = anchor
            forced = -first_dir
            it += 1
            continue
        if curv < stop_curvature and abs(gpar) < stop_parallel:
            reached = True
            break
        direction = forced if forced is not None else (1.0 if gpar >= 0.0 else -1.0)
        if forced is not None and abs(gpar) > 1e-3:
            forced = None
            direction = 1.0 if gpar >= 0.0 else -1.0
        step_coeff = np.zeros(ev_i.size)
        step_coeff[k] = step * direction * align
        others = [j for j in range(ev_i.size) if j != k]
        perp = np.empty(len(others))
        for n, j in enumerate(others):
            curv_j = float(ev_i[j])
            denom = curv_j if curv_j < 0.0 or curv_j >= curvature_floor else curvature_floor
            perp[n] = -float(gf @ vecs_i[:, j]) / denom
        pnorm = float(np.linalg.norm(perp))
        if pnorm > perp_cap:
            perp *= perp_cap / pnorm
        step_coeff[others] = perp
        x = x + basis @ (vecs_i @ step_coeff)
        it += 1
    if not reached or last_g is None or last_h is None:
        ev = np.linalg.eigvalsh(last_h) if last_h is not None else np.zeros(1)
        neg = int(np.sum(ev < NEG_CUT))
        gn = float(np.linalg.norm(last_g)) if last_g is not None else 1e9
        return RefineResult(
            x=x,
            calls=force + hess,
            force_calls=force,
            hessian_calls=hess,
            converged=False,
            index=neg,
            gradient_norm=gn,
        )
    energy, g = energy_grad(x)
    force += 1
    u, h = projected_hessian(x)
    hess += 1
    delta = finish_delta
    for _ in range(finish_steps):
        gn = float(np.linalg.norm(g))
        ev, vecs = np.linalg.eigh(h)
        neg = int(np.sum(ev < NEG_CUT))
        if gn < FORCE_TOL and neg == 1:
            return RefineResult(
                x=np.ascontiguousarray(x, dtype=np.float64).copy(),
                calls=force + hess,
                force_calls=force,
                hessian_calls=hess,
                converged=True,
                index=neg,
                gradient_norm=gn,
            )
        gf = u.T @ g
        trial = u @ prfo_step(ev, vecs, gf, delta)
        nrm = float(np.linalg.norm(trial))
        if nrm < 1e-14:
            break
        pred = float(g @ trial + 0.5 * trial @ (u @ (h @ (u.T @ trial))))
        e1, g1 = energy_grad(x + trial)
        force += 1
        g1n = float(np.linalg.norm(g1))
        rho = (e1 - energy) / pred if abs(pred) >= 1e-14 else 1.0
        if not (0.2 <= rho <= 5.0 or g1n < gn):
            delta = max(min(delta, nrm) * 0.5, 1e-3)
            continue
        u1, h1 = projected_hessian(x + trial)
        hess += 1
        ev1 = np.linalg.eigvalsh(h1)
        neg1 = int(np.sum(ev1 < NEG_CUT))
        if neg1 != 1 or g1n > gn * 1.05:
            delta = max(min(delta, nrm) * 0.5, 1e-3)
            continue
        x = x + trial
        energy, g, u, h = e1, g1, u1, h1
        if 0.7 < rho < 1.3 and nrm > 0.5 * delta:
            delta = min(delta * 1.3, 0.8)
    gn = float(np.linalg.norm(g))
    ev = np.linalg.eigvalsh(h)
    neg = int(np.sum(ev < NEG_CUT))
    return RefineResult(
        x=np.ascontiguousarray(x, dtype=np.float64).copy(),
        calls=force + hess,
        force_calls=force,
        hessian_calls=hess,
        converged=gn < FORCE_TOL and neg == 1,
        index=neg,
        gradient_norm=gn,
    )
