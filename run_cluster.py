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
from math import comb
import warnings
import numpy as np
import cvxpy as cp
import mosek

def werner_eve_states(W):
    """
    Generate Eve conditional states for a Werner state.

    Returns:
        M0, M1 : 2x2 numpy arrays
    """
    if not np.isfinite(W) or not 0 <= W <= 1:
        raise ValueError('Require 0 <= W <= 1.')
    a = (1 + 3 * W) / 4
    b = (1 - W) / 4
    M0 = np.array([[a, np.sqrt(a * b)], [np.sqrt(a * b), b]])
    M1 = b * np.ones((2, 2))
    return (M0, M1)

def generate_sector_states(W, n):
    """
    Generate the n+1 symmetry sector states.

    R_r corresponds to r copies of M1 and n-r copies of M0.
    """
    M0, M1 = werner_eve_states(W)
    states = []
    for r in range(n + 1):
        state = np.ones((1, 1))
        for sector in [0] * (n - r) + [1] * r:
            if sector == 0:
                state = np.kron(state, M0)
            else:
                state = np.kron(state, M1)
        states.append(state)
    return states
'Exact symmetry reduction of the supplied Werner CQ collision-entropy SDP.\n'

def solve_werner_reduced(W=0.85, epsilon=0.001, n=1, solver='MOSEK', verbose=False, max_n=100, tol=1e-05):
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
        for col_idx in range(d):
            constraints.append(cp.SOC(K + 1, cp.hstack([2 * ar[:, col_idx], K - 1])))
    overlap = sum((comb(n, r) * cp.trace(ar @ rr) for r, (ar, rr) in enumerate(zip(A, R))))
    problem = cp.Problem(cp.Maximize(2 * overlap - 2 * epsilon * T - K), constraints)
    solver = solver.upper()
    options = {}
    if solver == 'CLARABEL':
        options = dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=300)
    elif solver == 'SCS':
        options = dict(eps=tol, max_iters=200000)
    elif solver == 'MOSEK':
        options = dict(mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol})
    problem.solve(solver=solver, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND, **options)
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
    return dict(q=scaled_q / d, H_bits=n - np.log2(scaled_q), status=problem.status, A=values, T=tv, K=kv, solve_time=problem.solver_stats.solve_time, problem=problem)

def print_comparison(max_n):
    print(' n   baseline matrix vars   reduced matrix vars   baseline/reduced max PSD order')
    for n in range(1, max_n + 1):
        d = 2 ** n
        original = d * (d * d) * (d * d + 1)
        reduced = (n + 1) * d * (d + 1) // 2
        print(f'{n:2d} {original:22,d} {reduced:22,d} {2 * d * d:14,d} / {d:,}')
import gc
W = 0.85
epsilon = 0.001
n_max = 5
for n in range(1, n_max + 1):
    res = solve_werner_reduced(W=W, epsilon=epsilon, n=n, solver='MOSEK', tol=1e-05)
    print(f"H2 ({n} copies): {res['H_bits']:.6f} bits | {res['H_bits'] / n:.6f} bits/copy | time: {res['solve_time']:.4f} s | Q*: {res['q']:.8f}")
    del res
    gc.collect()
import numpy as np
import cvxpy as cp
import mosek

def sanity_measured_smooth_collision_entropy(rho_blocks, epsilon=0.001, verbose=False):
    """
    Compute H_2^{epsilon,M,up}(X|E) using Eq. (B8).
    rho_blocks[x] = p(x) rho_{E|x}.
    """
    nX = len(rho_blocks)
    dE = rho_blocks[0].shape[0]
    I = np.eye(dE)
    B = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    C = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    t = cp.Variable()
    k = cp.Variable()
    constraints = []
    for x in range(nX):
        constraints += [B[x] >> 0, B[x] << t * I, cp.bmat([[C[x], B[x]], [B[x], I]]) >> 0]
    constraints += [sum(C) << k * I]
    trace_term = sum((cp.real(cp.trace(B[x] @ rho_blocks[x])) for x in range(nX)))
    objective = cp.Maximize(2 * trace_term - 2 * epsilon * t - k)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND)
    Q_star = float(problem.value)
    H2 = -np.log2(Q_star)
    return {'entropy': H2, 'Q_star': Q_star, 't': t.value, 'k': k.value, 'status': problem.status}
import numpy as np

def partial_trace_AB(rho_ABE):
    """
    Traces out subsystems A (dim=2) and B (dim=2) from rho_ABE (dim=16x16),
    returning the reduced density matrix on E (dim=4x4).
    """
    rho_tensor = rho_ABE.reshape(2, 2, 4, 2, 2, 4)
    return np.einsum('ijkijl->kl', rho_tensor)

def generate_werner_cq_blocks(W, n=1):
    """
    Generate CQ blocks tau_x = p(x) rho_{E|x} for n copies of Werner state with visibility W.
    Returns 2^n subnormalized blocks of size (4^n, 4^n).
    """
    z0 = np.array([1, 0], dtype=complex)
    z1 = np.array([0, 1], dtype=complex)
    phi_plus = (np.kron(z0, z0) + np.kron(z1, z1)) / np.sqrt(2)
    phi_minus = (np.kron(z0, z0) - np.kron(z1, z1)) / np.sqrt(2)
    psi_plus = (np.kron(z0, z1) + np.kron(z1, z0)) / np.sqrt(2)
    psi_minus = (np.kron(z0, z1) - np.kron(z1, z0)) / np.sqrt(2)
    bell_basis = [phi_plus, phi_minus, psi_plus, psi_minus]
    lam = np.array([(1.0 + 3.0 * W) / 4.0, (1.0 - W) / 4.0, (1.0 - W) / 4.0, (1.0 - W) / 4.0], dtype=float)
    psi_ABE = np.zeros(2 * 2 * 4, dtype=complex)
    for k in range(4):
        e_k = np.zeros(4, dtype=complex)
        e_k[k] = 1.0
        psi_ABE += np.sqrt(lam[k]) * np.kron(bell_basis[k], e_k)
    rho_ABE = np.outer(psi_ABE, psi_ABE.conj())
    tau_single = []
    proj_A = [np.outer(z0, z0.conj()), np.outer(z1, z1.conj())]
    I_B = np.eye(2, dtype=complex)
    I_E = np.eye(4, dtype=complex)
    for x in range(2):
        proj_meas = np.kron(np.kron(proj_A[x], I_B), I_E)
        rho_post = proj_meas @ rho_ABE @ proj_meas
        tau_x = partial_trace_AB(rho_post)
        tau_single.append(tau_x)
    blocks = tau_single
    for _ in range(1, n):
        blocks = [np.kron(b, t) for b in blocks for t in tau_single]
    return blocks
test_max = 2
for test in range(1, test_max + 1):
    blocks = generate_werner_cq_blocks(W=W, n=test)
    res = sanity_measured_smooth_collision_entropy(rho_blocks=blocks, epsilon=0.001)
    print(f"Sanity H2 ({test} copies): {res['entropy']:.6f} bits | {res['entropy'] / test:.6f} bits/copy | Q*: {res['Q_star']:.8f}")
