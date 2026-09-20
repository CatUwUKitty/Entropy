import os
import sys
from pathlib import Path
import numpy as np
import mosek
import mosek.fusion as mf

def setup_mosek_license():
    """Ensure MOSEK can locate a valid license file on the cluster."""
    if "MOSEKLM_LICENSE_FILE" not in os.environ:
        default_paths = [
            Path.home() / "mosek" / "mosek.lic",
            Path.home() / ".mosek" / "mosek.lic",
        ]
        for p in default_paths:
            if p.exists():
                os.environ["MOSEKLM_LICENSE_FILE"] = str(p)
                break

setup_mosek_license()


import os
import sys
from pathlib import Path
'Exact symmetry reduction of the supplied Werner CQ collision-entropy SDP.\n'
from math import comb
import warnings
import numpy as np
import cvxpy as cp
import mosek

def solve_werner_reduced(W=0.85, epsilon=0.001, n=1, solver='CLARABEL', verbose=False, max_n=100, tol=1e-07):
    """Return q, entropy, residuals and n+1 real symmetric sector matrices.

    A_r = 2**n B_{0^n,r}; T = 2**n t; K = 2**n k.
    Each A_r has dimension 2**n. No C matrices or Schur PSD blocks.
    One global epsilon is used. max_n is an allocation guard, not a promise
    that every permitted n will solve quickly on the available hardware.
    """
    if not np.isfinite(W) or not 0 <= W <= 1:
        raise ValueError('Require 0 <= W <= 1.')
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError('Require 0 <= epsilon < 1.')
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer.')
    if n > max_n:
        raise ValueError(f'n={n} exceeds max_n={max_n}; assess scaling first.')
    if tol <= 0:
        raise ValueError('tol must be positive.')
    a, b = ((1 + 3 * W) / 4, (1 - W) / 4)
    M = [np.array([[a, np.sqrt(a * b)], [np.sqrt(a * b), b]]), b * np.ones((2, 2))]
    d = 2 ** n
    I = np.eye(d)
    A = [cp.Variable((d, d), symmetric=True, name=f'A_{r}') for r in range(n + 1)]
    T = cp.Variable(nonneg=True, name='T')
    K = cp.Variable(nonneg=True, name='K')
    constraints, R = ([], [])
    for r, ar in enumerate(A):
        rr = np.ones((1, 1))
        for sector in [0] * (n - r) + [1] * r:
            rr = np.kron(rr, M[sector])
        R.append(rr)
        constraints += [ar >> 0, T * I - ar >> 0]
        tails = cp.reshape(K - 1, (1, 1), order='C') @ np.ones((1, d))
        constraints.append(cp.SOC((K + 1) * np.ones(d), cp.vstack([2 * ar, tails]), axis=0))
    overlap = sum((comb(n, r) * cp.trace(ar @ rr) for r, (ar, rr) in enumerate(zip(A, R))))
    problem = cp.Problem(cp.Maximize(2 * overlap - 2 * epsilon * T - K), constraints)
    solver = solver.upper()
    options = {}
    if solver == 'CLARABEL':
        options = dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=300)
    elif solver == 'SCS':
        options = dict(eps=tol, max_iters=200000)
    problem.solve(solver=solver, verbose=verbose, **options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'Solver status: {problem.status}')
    if problem.status == cp.OPTIMAL_INACCURATE:
        warnings.warn('optimal_inaccurate: cross-check tolerances or another solver.')
    values = [(ar.value + ar.value.T) / 2 for ar in A]
    tv, kv = (float(T.value), float(K.value))
    psd_violation = max([0.0, -tv, -kv] + [-float(np.linalg.eigvalsh(ar).min()) for ar in values] + [-float(np.linalg.eigvalsh(tv * I - ar).min()) for ar in values])
    norm_violation = max(0.0, max((float(np.max(np.sum(ar ** 2, axis=0) - kv)) for ar in values)))
    if max(psd_violation, norm_violation) > 100 * tol:
        warnings.warn('Constraint residual exceeds 100*tol; inspect returned residuals.')
    scaled_q = float(problem.value)
    if not np.isfinite(scaled_q) or scaled_q <= 0:
        raise RuntimeError('Nonpositive/nonfinite optimum; cannot reliably compute entropy.')
    return dict(q=scaled_q / d, H_bits=n - np.log2(scaled_q), status=problem.status, A=values, T=tv, K=kv, psd_violation=psd_violation, column_squared_norm_violation=norm_violation, solve_time=problem.solver_stats.solve_time, problem=problem)

def print_comparison(max_n):
    print(' n   baseline matrix vars   reduced matrix vars   baseline/reduced max PSD order')
    for n in range(1, max_n + 1):
        d = 2 ** n
        original = d * (d * d) * (d * d + 1)
        reduced = (n + 1) * d * (d + 1) // 2
        print(f'{n:2d} {original:22,d} {reduced:22,d} {2 * d * d:14,d} / {d:,}')
W = 0.85
epsilon = 0.001
n = 7
result = solve_werner_reduced(W=W, epsilon=epsilon, n=n, solver='MOSEK', tol=1e-07)
print('status:', result['status'])
print('q_epsilon:', result['q'])
print('entropy [bits]:', result['H_bits'])
print('entropy per copy:', result['H_bits'] / n)
print('solver time [s]:', result['solve_time'])
print('scaled PSD violation:', result['psd_violation'])
print('scaled squared-column-norm violation:', result['column_squared_norm_violation'])
del result
print_comparison(10)
import gc
gc.collect()
