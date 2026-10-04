import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from math import comb, lgamma
import numpy as np
import pandas as pd
import cvxpy as cp

IN_COLAB = False
if 'MOSEKLM_LICENSE_FILE' not in os.environ:
    for candidate in (Path.home() / 'mosek' / 'mosek.lic', Path.home() / '.mosek' / 'mosek.lic'):
        if candidate.exists():
            os.environ['MOSEKLM_LICENSE_FILE'] = str(candidate)
            break

DEFAULT_CALCULATIONS = ['collision', 'min_entropy', 'classical']

# @title **02. Shared Parameters and Solver Settings**
# Parameter sweeps: add values to any list. All combinations are evaluated.
W_VALUES = [0.80, 0.85, 0.9, 0.95, 1.0]                         # Example: [0.4, 0.7, 0.85]
COLLISION_EPSILON_VALUES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]          # Example: [1e-4, 1e-3, 1e-2]
MIN_ENTROPY_EPSILON_VALUES = list(COLLISION_EPSILON_VALUES)
COLLISION_N_VALUES = list(range(1, 7))      # Example: list(range(1, 10))
MIN_ENTROPY_N_VALUES = list(range(1, 7))

# Classical Eve can use entropy-matched Werner cases, fixed error rates, or both.
CLASSICAL_MODES = ['matched', 'fixed']
CLASSICAL_P_ERR_VALUES = [0.1]
CLASSICAL_N_VALUES = [1, 2, 5, 10, 20, 50, 100, 1000, 10000]
CLASSICAL_PLOT_MAX_N = None               # Example: 15, 100, or 10000

COLLISION_SOLVER = 'MOSEK'                # Alternatives: 'CLARABEL', 'SCS'
MIN_ENTROPY_SOLVER = 'MOSEK'
COLLISION_TOL = 1e-5
MIN_ENTROPY_TOL = 1e-5
MAX_QUANTUM_COPIES = 10                  # Allocation guard; large n can be costly.
SOLVER_VERBOSE = False
RESUME = True
CONTINUE_ON_ERROR = True
RUN_VALIDATION = True
VALIDATION_N_VALUES = [1, 2]
VALIDATION_RTOL = 1e-4
VALIDATION_ATOL = 1e-7

OUTPUT_DIR = (Path('/content/drive/MyDrive/werner_entropy_results')
              if IN_COLAB else Path('werner_entropy_results'))
MODEL_VERSION = 'combined-werner-v1'


def validate_parameters(W, n, epsilon=None, min_entropy=False):
    if not np.isfinite(W) or not 0 <= W <= 1:
        raise ValueError('Require 0 <= W <= 1.')
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer.')
    if epsilon is not None and (not np.isfinite(epsilon) or not 0 <= epsilon < 1):
        raise ValueError('Require 0 <= smoothing parameter < 1.')
    if n > MAX_QUANTUM_COPIES:
        raise ValueError(f'n={n} exceeds MAX_QUANTUM_COPIES={MAX_QUANTUM_COPIES}.')


def checked_values(values, name, kind='probability'):
    values = list(values)
    if not values:
        raise ValueError(f'{name} must contain at least one value.')
    for value in values:
        if kind == 'copies':
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must contain positive integers.')
        elif not np.isfinite(value) or not 0 <= value <= 1 or (kind == 'epsilon' and value == 1):
            raise ValueError(f'Invalid value {value!r} in {name}.')
    return list(dict.fromkeys(values))


def quantum_grid(epsilon_values, n_values, solver, tol):
    ws = checked_values(W_VALUES, 'W_VALUES')
    es = checked_values(epsilon_values, 'epsilon values', 'epsilon')
    ns = checked_values(n_values, 'copy counts', 'copies')
    for W, epsilon, n in itertools.product(ws, es, ns):
        validate_parameters(W, n, epsilon)
        yield dict(W=float(W), epsilon=float(epsilon), n=int(n),
                   solver=solver.upper(), tol=float(tol))


def require_solver(solver):
    if solver.upper() not in cp.installed_solvers():
        raise RuntimeError(f'Solver {solver} is unavailable: {cp.installed_solvers()}')


def solver_options(solver, tol):
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite.')
    solver = solver.upper()
    if solver == 'CLARABEL':
        return dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=500)
    if solver == 'SCS':
        return dict(eps=tol, max_iters=200000)
    if solver == 'MOSEK':
        return dict(mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol})
    return {}


def checkpoint_key(calculation, case):
    payload = dict(model_version=MODEL_VERSION, calculation=calculation, **case)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_checkpoint(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + '.tmp')
    pd.DataFrame(list(records.values())).to_csv(temporary_path, index=False, float_format='%.17g')
    os.replace(temporary_path, path)


def run_sweep(calculation, grid, solve_case):
    cases = list(grid)
    if not cases:
        raise ValueError(f'{calculation}: the parameter grid is empty.')
    path = Path(OUTPUT_DIR) / f'{calculation}_results.csv'
    records = {}
    if RESUME and path.exists():
        previous = pd.read_csv(path, float_precision='round_trip')
        if 'case_key' not in previous or 'run_status' not in previous:
            raise ValueError(f'Unrecognized checkpoint schema: {path}')
        if 'error' in previous:
            previous['error'] = previous['error'].fillna('')
        records = {row['case_key']: row for row in previous.to_dict('records')}
    current_rows = []
    for index, case in enumerate(cases, 1):
        key = checkpoint_key(calculation, case)
        if key in records and records[key].get('run_status') == 'ok':
            row = records[key]
            print(f'[{index}/{len(cases)}] {calculation}: reused {case}')
        else:
            print(f'[{index}/{len(cases)}] {calculation}: solving {case}', flush=True)
            started = time.perf_counter()
            row = dict(case_key=key, model_version=MODEL_VERSION, **case)
            try:
                row.update(solve_case(case))
                row.update(run_status='ok', error='', wall_seconds=time.perf_counter() - started)
            except Exception as error:
                row.update(run_status='error', error=f'{type(error).__name__}: {error}',
                           wall_seconds=time.perf_counter() - started)
                records[key] = row
                write_checkpoint(records, path)
                print(f"  Failed: {row['error']}", flush=True)
                if not CONTINUE_ON_ERROR:
                    raise
            records[key] = row
            write_checkpoint(records, path)
        current_rows.append(row)
    frame = pd.DataFrame(current_rows)
    print(f"{calculation}: {(frame['run_status'] == 'ok').sum()}/{len(frame)} successful; {path}")
    return frame


def successful_results(frame, name):
    if frame is None or frame.empty:
        print(f'{name}: no results available.')
        return pd.DataFrame()
    good = frame.loc[frame['run_status'] == 'ok'].copy()
    if good.empty:
        print(f'{name}: no successful results. Inspect the error column in its CSV.')
    return good


def print_result_table(frame, columns):
    selected = [column for column in columns if column in frame.columns]
    selected += [column for column in ('run_status', 'error') if column in frame.columns]
    print(frame[selected].to_string(index=False))


# @title **03. Werner State and Symmetry-Sector Generation**
def werner_eve_states(W):
    """
    Generate Eve conditional states for a Werner state.

    Returns:
        M0, M1 : 2x2 numpy arrays
    """
    if not np.isfinite(W) or not 0 <= W <= 1:
        raise ValueError("Require 0 <= W <= 1.")

    a = (1 + 3*W) / 4
    b = (1 - W) / 4

    M0 = np.array([
        [a, np.sqrt(a*b)],
        [np.sqrt(a*b), b]
    ])

    M1 = b * np.ones((2, 2))

    return M0, M1

def generate_sector_states(W, n):
    """
    Generate the n+1 symmetry sector states.

    R_r corresponds to r copies of M1 and n-r copies of M0.
    """
    validate_parameters(W, n)
    M0, M1 = werner_eve_states(W)

    states = []

    for r in range(n+1):
        state = np.ones((1, 1))

        # Any ordering works because of tensor symmetry
        for sector in [0]*(n-r) + [1]*r:
            if sector == 0:
                state = np.kron(state, M0)
            else:
                state = np.kron(state, M1)

        states.append(state)

    return states

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
    validate_parameters(W, n)
    # 1. Computational basis
    z0 = np.array([1, 0], dtype=complex)
    z1 = np.array([0, 1], dtype=complex)

    # 2. Bell states
    phi_plus  = (np.kron(z0, z0) + np.kron(z1, z1)) / np.sqrt(2)
    phi_minus = (np.kron(z0, z0) - np.kron(z1, z1)) / np.sqrt(2)
    psi_plus  = (np.kron(z0, z1) + np.kron(z1, z0)) / np.sqrt(2)
    psi_minus = (np.kron(z0, z1) - np.kron(z1, z0)) / np.sqrt(2)
    bell_basis = [phi_plus, phi_minus, psi_plus, psi_minus]

    # 3. Werner state eigenvalues in Bell basis
    lam = np.array([
        (1.0 + 3.0 * W) / 4.0,
        (1.0 - W) / 4.0,
        (1.0 - W) / 4.0,
        (1.0 - W) / 4.0
    ], dtype=float)

    # 4. Build Purification |Psi>_{ABE} with d_E = 4
    psi_ABE = np.zeros(2 * 2 * 4, dtype=complex)
    for k in range(4):
        e_k = np.zeros(4, dtype=complex)
        e_k[k] = 1.0
        psi_ABE += np.sqrt(lam[k]) * np.kron(bell_basis[k], e_k)

    rho_ABE = np.outer(psi_ABE, psi_ABE.conj())

    # 5. Project Alice onto |0><0| and |1><1|, then trace out A and B
    tau_single = []
    proj_A = [np.outer(z0, z0.conj()), np.outer(z1, z1.conj())]
    I_B = np.eye(2, dtype=complex)
    I_E = np.eye(4, dtype=complex)

    for x in range(2):
        proj_meas = np.kron(np.kron(proj_A[x], I_B), I_E)
        rho_post = proj_meas @ rho_ABE @ proj_meas
        tau_x = partial_trace_AB(rho_post)
        tau_single.append(tau_x)

    # 6. Accumulate Kronecker products for n copies
    blocks = tau_single
    for _ in range(1, n):
        blocks = [np.kron(b, t) for b in blocks for t in tau_single]

    return blocks


# @title **04. Collision Entropy: Reduced SDP Solver**
def solve_werner_reduced(W=0.85, epsilon=1e-3, n=1, solver="MOSEK",
                         verbose=False, max_n=None, tol=1e-5):
    """Return q, entropy, residuals and n+1 real symmetric sector matrices.

    A_r = 2**n B_{0^n,r}; T = 2**n t; K = 2**n k.
    Each A_r has dimension 2**n. No C matrices or Schur PSD blocks.
    One global epsilon is used. max_n is an allocation guard, not a promise
    that every permitted n will solve quickly on the available hardware.
    """
    validate_parameters(W, n, epsilon)
    require_solver(solver)
    if not np.isfinite(W) or not 0 <= W <= 1:
        raise ValueError("Require 0 <= W <= 1.")
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("Require 0 <= epsilon < 1.")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    max_n = MAX_QUANTUM_COPIES if max_n is None else max_n
    if n > max_n:
        raise ValueError(f"n={n} exceeds max_n={max_n}; assess scaling first.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    sector_inputs = generate_sector_states(W, n)
    d = 2**n
    I = np.eye(d)
    A = [cp.Variable((d, d), symmetric=True, name=f"A_{r}") for r in range(n+1)]
    T = cp.Variable(nonneg=True, name="T")
    K = cp.Variable(nonneg=True, name="K")
    constraints, R = [], []
    for r, ar in enumerate(A):
        rr = sector_inputs[r]
        R.append(rr)
        constraints += [ar >> 0, T*I-ar >> 0]
        # Each squared column norm <= K. A rotated SOC written as an SOC:
        # ||[2*a_col, K-1]||_2 <= K+1  <=>  ||a_col||_2^2 <= K.
        # Vectorized: one cone per column
        # Rotate SOC: ||col||_2^2 <= K <=> || [2*col, K-1] ||_2 <= K+1
        for col_idx in range(d):
            constraints.append(
            cp.SOC(K + 1, cp.hstack([2 * ar[:, col_idx], K - 1]))
        )
    overlap = sum(comb(n, r)*cp.trace(ar @ rr) for r, (ar, rr) in enumerate(zip(A, R)))
    # Optimize scaled q to avoid an exponentially small objective.
    problem = cp.Problem(cp.Maximize(2*overlap - 2*epsilon*T - K), constraints)
    solver = solver.upper()
    options = solver_options(solver, tol)

    problem.solve(solver=solver, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND, **options)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver status: {problem.status}")
    if problem.status == cp.OPTIMAL_INACCURATE:
        warnings.warn("optimal_inaccurate: cross-check tolerances or another solver.")

    values = [(ar.value + ar.value.T)/2 for ar in A]
    tv, kv = float(T.value), float(K.value)
    psd_violation = max([0., -tv, -kv] +
                        [-float(np.linalg.eigvalsh(ar).min()) for ar in values] +
                        [-float(np.linalg.eigvalsh(tv*I-ar).min()) for ar in values])
    norm_violation = max(0., max(float(np.max(np.sum(ar**2, axis=0)-kv)) for ar in values))

    # These residuals are in scaled variables, not baseline units.
    if max(psd_violation, norm_violation) > 100*tol:
        warnings.warn("Constraint residual exceeds 100*tol; inspect returned residuals.")
    scaled_q = float(problem.value)
    if not np.isfinite(scaled_q) or scaled_q <= 0:
        raise RuntimeError("Nonpositive/nonfinite optimum; cannot reliably compute entropy.")

    return dict(q=scaled_q/d, H_bits=n-np.log2(scaled_q), status=problem.status,
                A=values, T=tv, K=kv, solve_time=problem.solver_stats.solve_time, problem=problem,
                psd_violation=psd_violation, norm_violation=norm_violation)

def print_comparison(max_n):
    print(" n   baseline matrix vars   reduced matrix vars   baseline/reduced max PSD order")
    for n in range(1, max_n+1):
        d = 2**n
        original = d*(d*d)*(d*d+1)
        reduced = (n+1)*d*(d+1)//2
        print(f"{n:2d} {original:22,d} {reduced:22,d} {2*d*d:14,d} / {d:,}")


# @title **05. Collision Entropy: Baseline SDP Solver**
def sanity_measured_smooth_collision_entropy(
    rho_blocks,
    epsilon=0.001,
    verbose=False,
    solver="MOSEK",
    tol=1e-8,
):

    """
    Compute H_2^{epsilon,M,up}(X|E) using Eq. (B8).
    rho_blocks[x] = p(x) rho_{E|x}.
    """

    require_solver(solver)
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("Require 0 <= epsilon < 1.")
    nX = len(rho_blocks)
    dE = rho_blocks[0].shape[0]
    I = np.eye(dE)

    # Variables
    B = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    C = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    t = cp.Variable()
    k = cp.Variable()

    # Constraints
    constraints = []
    for x in range(nX):
        constraints += [
            B[x] >> 0,
            B[x] << t * I,
            cp.bmat([
                [C[x], B[x]],
                [B[x], I],
            ]) >> 0,
        ]

    constraints += [
        sum(C) << k * I
    ]

    # Objective
    trace_term = sum(
        cp.real(cp.trace(B[x] @ rho_blocks[x]))
        for x in range(nX)
    )

    objective = cp.Maximize(
        2 * trace_term - 2 * epsilon * t - k
    )

    problem = cp.Problem(objective, constraints)

    # Solve
    problem.solve(
        solver=solver.upper(),
        verbose=verbose,
        canon_backend=cp.SCIPY_CANON_BACKEND,
        **solver_options(solver, tol),
    )


    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver status: {problem.status}")
    Q_star = float(problem.value)
    if not np.isfinite(Q_star) or Q_star <= 0:
        raise RuntimeError("Cannot compute entropy from a nonpositive/nonfinite optimum.")
    H2 = -np.log2(Q_star)

    return {
        "entropy": H2,
        "Q_star": Q_star,
        "t": t.value,
        "k": k.value,
        "status": problem.status,
    }


def collision_case(case):
    result = solve_werner_reduced(W=case['W'], epsilon=case['epsilon'], n=case['n'], solver=case['solver'], tol=case['tol'], verbose=SOLVER_VERBOSE)
    return dict(q=result['q'], H_bits=result['H_bits'], bits_per_copy=result['H_bits'] / case['n'], solver_status=result['status'], solve_seconds=result['solve_time'], psd_violation=result['psd_violation'], norm_violation=result['norm_violation'])

# @title **08. Min-Entropy: Reduced SDP Solver**
def solve_min_entropy_reduced(W, n, smooth_radius_squared, solver="MOSEK", tol=1e-9, verbose=False):
    """Return min Tr(S_B) for simple.py's _solve_smooth_alt_sdp.

    Here smooth_radius_squared is the epsilon passed to the square root in
    simple.py: c = sqrt(1 - epsilon).
    """
    validate_parameters(W, n, smooth_radius_squared, min_entropy=True)
    require_solver(solver)
    solver = solver.upper()
    d, m = 2**n, 2**n
    R = generate_sector_states(W, n)
    U = [cp.Variable((d, d), symmetric=True) for _ in R]
    V = [cp.Variable((d, d)) for _ in R]
    q = [cp.Variable(d, nonneg=True) for _ in R]
    weights = [math.comb(n, r) for r in range(n + 1)]
    constraints = []
    for rr, uu, vv, qq in zip(R, U, V, q):
        constraints += [cp.diag(qq) - uu >> 0]
        constraints += [cp.bmat([[rr, vv], [vv.T, uu]]) >> 0]
    constraints += [
        sum(w * cp.trace(uu) for w, uu in zip(weights, U)) <= 1,
        sum(w * cp.trace(vv) for w, vv in zip(weights, V))
        >= math.sqrt(1 - smooth_radius_squared),
    ]
    problem = cp.Problem(
        cp.Minimize(sum(w * cp.sum(qq) for w, qq in zip(weights, q)) / m),
        constraints,
    )
    start = time.perf_counter()
    problem.solve(solver=solver, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND, **solver_options(solver, tol))
    elapsed = time.perf_counter() - start
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"n={n}, radius^2={smooth_radius_squared}: {problem.status}")
    if problem.status == cp.OPTIMAL_INACCURATE:
        warnings.warn("Min-entropy solve returned optimal_inaccurate; check numerical accuracy.")
    value = float(problem.value)
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("Cannot compute min-entropy from a nonpositive/nonfinite optimum.")
    return value, problem.status, elapsed


# @title **09. Min-Entropy: Baseline SDP Solver**
def solve_min_entropy_baseline(W, n, smooth_radius_squared, solver="MOSEK", tol=1e-9, verbose=False):
    """Direct transcription of simple.py for cross-checks at small n."""
    validate_parameters(W, n, smooth_radius_squared, min_entropy=True)
    require_solver(solver)
    solver = solver.upper()
    blocks = [np.real_if_close(block) for block in generate_werner_cq_blocks(W, n)]
    d = 4**n
    S = cp.Variable((d, d), symmetric=True)
    taus = [cp.Variable((d, d), symmetric=True) for _ in blocks]
    Ls = [cp.Variable((d, d)) for _ in blocks]
    constraints = [S >> 0]
    for B, tau, L in zip(blocks, taus, Ls):
        constraints += [S - tau >> 0, cp.bmat([[B, L], [L.T, tau]]) >> 0]
    constraints += [
        sum(cp.trace(t) for t in taus) <= 1,
        sum(cp.trace(L) for L in Ls) >= math.sqrt(1 - smooth_radius_squared),
    ]
    problem = cp.Problem(cp.Minimize(cp.trace(S)), constraints)
    problem.solve(solver=solver, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND, **solver_options(solver, tol))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver status: {problem.status}")
    value = float(problem.value)
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("Nonpositive/nonfinite min-entropy optimum.")
    return value, problem.status


def min_entropy_case(case):
    epsilon = case['epsilon']
    mu = epsilon / 2
    low_v, low_status, low_seconds = solve_min_entropy_reduced(case['W'], case['n'], epsilon / 2, case['solver'], case['tol'], SOLVER_VERBOSE)
    up_v, up_status, up_seconds = solve_min_entropy_reduced(case['W'], case['n'], epsilon, case['solver'], case['tol'], SOLVER_VERBOSE)
    H_low, H_up = (-math.log2(low_v), -math.log2(up_v))
    ell_low = max(0, math.floor(H_low + math.log2(4 * mu ** 3))) if mu > 0 else np.nan
    ell_up = max(0, math.floor(H_up + math.log2(1 / (1 - epsilon)))) if epsilon > 0 else np.nan
    return dict(v_low=low_v, v_up=up_v, H_alt_low=H_low, H_alt_up=H_up, alt_low=ell_low, alt_up=ell_up, bits_per_copy=H_low / case['n'], bits_per_copy_up=H_up / case['n'], radius_low=math.sqrt(epsilon / 2), radius_up=math.sqrt(epsilon), radius_squared_low=epsilon / 2, radius_squared_up=epsilon, low_status=low_status, up_status=up_status, low_seconds=low_seconds, up_seconds=up_seconds)

# @title **12. Classical Eve: State Generation and Collision Entropy Solver**
def _logsumexp(a):
    largest = np.max(a)
    return float(largest + np.log(np.exp(a-largest).sum()))

def generate_noisy_bit_cq_blocks(p_err, n=1):
    """Small-n input to sanity_measured_smooth_collision_entropy."""
    if not np.isfinite(p_err) or not 0 <= p_err <= 1:
        raise ValueError("Require 0 <= p_err <= 1.")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    single = [0.5*np.diag([1-p_err, p_err]),
              0.5*np.diag([p_err, 1-p_err])]
    blocks = single
    for _ in range(n-1):
        blocks = [np.kron(b, s) for b in blocks for s in single]
    return blocks

def solve_noisy_bit_reduced(p_err=0.1, epsilon=1e-3, n=1):
    """Exact scalar reduction of the notebook's B8 SDP; floating-point solve.

    Uses one global epsilon, O(n) storage, and log-domain calculations.
    q may underflow at large n, but H_bits is computed directly from log_q.
    """
    start = time.perf_counter()
    if not np.isfinite(p_err) or not 0 <= p_err <= 1:
        raise ValueError("Require 0 <= p_err <= 1.")
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("Require 0 <= epsilon < 1.")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")

    p = min(p_err, 1-p_err)  # Relabeling Eve's bit preserves the entropy.
    if p == 0:
        log_q = 2*np.log1p(-epsilon)
    elif p == 0.5:
        log_q = 2*np.log1p(-epsilon) - n*np.log(2.0)
    elif epsilon == 0:
        log_q = n*np.log((1-p)**2 + p**2)
    else:
        j = np.arange(n+1, dtype=float)
        log_c = np.array([lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
                          for k in range(n+1)])
        log_p = (n-j)*np.log1p(-p) + j*np.log(p)
        log_mass = log_c + log_p
        mass = np.exp(log_mass - _logsumexp(log_mass))

        def removed_mass(log_t):
            # Sum_j C(n,j) max(p_j-t, 0), evaluated without subtraction loss.
            fraction = -np.expm1(np.minimum(log_t-log_p, 0.0))
            return float(mass @ fraction)

        lo = float(log_p.min()+np.log1p(-epsilon)-1.0)
        hi = float(log_p.max())
        for _ in range(100):
            mid = lo + (hi-lo)/2
            if mid == lo or mid == hi:
                break
            if removed_mass(mid) > epsilon:
                lo = mid
            else:
                hi = mid
        log_t = lo + (hi-lo)/2
        log_q = _logsumexp(log_c + 2*np.minimum(log_p, log_t))

    return dict(q=float(np.exp(log_q)), log_q=float(log_q),
                H_bits=float(-log_q/np.log(2.0)),
                status="analytic" if p in (0, 0.5) or epsilon == 0 else "scalar_root",
                solve_time=time.perf_counter()-start)


def h2(p):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Require 0 <= p <= 1.')
    return -sum((float(t) * math.log2(float(t)) for t in (p, 1 - p) if t > 0))

def werner_conditional_entropy(W):
    if not np.isfinite(W) or not 0 <= W <= 1:
        raise ValueError('Require 0 <= W <= 1.')
    lam = [(1 + 3 * W) / 4] + [(1 - W) / 4] * 3
    return 1 + h2((1 - W) / 2) + sum((p * math.log2(p) for p in lam if p > 0))

def matched_error_probability(W):
    target = float(np.clip(werner_conditional_entropy(W), 0, 1))
    if target == 0:
        return (0.0, target)
    if target == 1:
        return (0.5, target)
    lo, hi = (0.0, 0.5)
    for _ in range(100):
        mid = (lo + hi) / 2
        if h2(mid) < target:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2, target)

def make_classical_scenarios():
    if not CLASSICAL_MODES or any((mode not in ('matched', 'fixed') for mode in CLASSICAL_MODES)):
        raise ValueError("CLASSICAL_MODES must contain 'matched', 'fixed', or both.")
    scenarios = []
    if 'matched' in CLASSICAL_MODES:
        for W in checked_values(W_VALUES, 'W_VALUES'):
            p, target = matched_error_probability(W)
            scenarios.append(dict(mode='matched', W=float(W), p_err=p, H_reference=target, label=f'matched W={W:g}'))
    if 'fixed' in CLASSICAL_MODES:
        for p in checked_values(CLASSICAL_P_ERR_VALUES, 'CLASSICAL_P_ERR_VALUES'):
            scenarios.append(dict(mode='fixed', W=None, p_err=float(p), H_reference=h2(p), label=f'fixed p_err={p:g}'))
    return scenarios

def classical_grid():
    for scenario, epsilon, n in itertools.product(classical_scenarios, checked_values(COLLISION_EPSILON_VALUES, 'collision epsilon', 'epsilon'), checked_values(CLASSICAL_N_VALUES, 'classical copy counts', 'copies')):
        yield dict(**scenario, epsilon=float(epsilon), n=int(n), solver='scalar_root')

def classical_case(case):
    result = solve_noisy_bit_reduced(p_err=case['p_err'], epsilon=case['epsilon'], n=case['n'])
    return dict(q=result['q'], log_q=result['log_q'], H_bits=result['H_bits'], bits_per_copy=result['H_bits'] / case['n'], solver_status=result['status'], solve_seconds=result['solve_time'])


def apply_cli_arguments(args):
    global W_VALUES, COLLISION_EPSILON_VALUES, MIN_ENTROPY_EPSILON_VALUES
    global COLLISION_N_VALUES, MIN_ENTROPY_N_VALUES, CLASSICAL_N_VALUES
    global CLASSICAL_MODES, CLASSICAL_P_ERR_VALUES
    global COLLISION_SOLVER, MIN_ENTROPY_SOLVER, COLLISION_TOL, MIN_ENTROPY_TOL
    global OUTPUT_DIR, RESUME, CONTINUE_ON_ERROR, MAX_QUANTUM_COPIES
    if args.W is not None:
        W_VALUES = args.W
    if args.collision_epsilon is not None:
        COLLISION_EPSILON_VALUES = args.collision_epsilon
        MIN_ENTROPY_EPSILON_VALUES = list(COLLISION_EPSILON_VALUES)
    if args.n is not None:
        COLLISION_N_VALUES = MIN_ENTROPY_N_VALUES = args.n
    if args.classical_n is not None:
        CLASSICAL_N_VALUES = args.classical_n
    if args.p_err is not None:
        CLASSICAL_P_ERR_VALUES = args.p_err
    if args.classical_modes is not None:
        CLASSICAL_MODES = args.classical_modes
    if args.solver is not None:
        COLLISION_SOLVER = MIN_ENTROPY_SOLVER = args.solver.upper()
    if args.tol is not None:
        COLLISION_TOL = MIN_ENTROPY_TOL = args.tol
    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)
    if args.max_quantum_copies is not None:
        MAX_QUANTUM_COPIES = args.max_quantum_copies
    RESUME = not args.no_resume
    CONTINUE_ON_ERROR = not args.fail_fast


def parse_arguments():
    parser = argparse.ArgumentParser(description='Checkpointed Werner entropy parameter sweeps.')
    parser.add_argument('--calculations', nargs='+', choices=['collision', 'min_entropy', 'classical'],
                        default=DEFAULT_CALCULATIONS)
    parser.add_argument('--W', nargs='+', type=float, help='Werner visibility values')
    parser.add_argument('--collision-epsilon', nargs='+', type=float,
                        help='Collision smoothing values; also override the min-entropy squared-radius values')
    parser.add_argument('--n', nargs='+', type=int, help='Quantum copy counts for both calculations')
    parser.add_argument('--classical-n', nargs='+', type=int)
    parser.add_argument('--p-err', nargs='+', type=float)
    parser.add_argument('--classical-modes', nargs='+', choices=['matched', 'fixed'])
    parser.add_argument('--solver', choices=['MOSEK', 'CLARABEL', 'SCS'])
    parser.add_argument('--tol', type=float)
    parser.add_argument('--max-quantum-copies', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    apply_cli_arguments(args)
    any_errors = False
    if 'collision' in args.calculations:
        collision_results = run_sweep('collision', quantum_grid(COLLISION_EPSILON_VALUES, COLLISION_N_VALUES, COLLISION_SOLVER, COLLISION_TOL), collision_case)
        print_result_table(collision_results, ['W', 'epsilon', 'n', 'H_bits', 'bits_per_copy', 'solver_status', 'solve_seconds'])
        if (collision_results['run_status'] == 'error').any():
            any_errors = True
    if 'min_entropy' in args.calculations:
        min_entropy_results = run_sweep('min_entropy', quantum_grid(MIN_ENTROPY_EPSILON_VALUES, MIN_ENTROPY_N_VALUES, MIN_ENTROPY_SOLVER, MIN_ENTROPY_TOL), min_entropy_case)
        print_result_table(min_entropy_results, ['W', 'epsilon', 'n', 'H_alt_low', 'H_alt_up', 'bits_per_copy', 'bits_per_copy_up', 'low_status', 'up_status'])
        if (min_entropy_results['run_status'] == 'error').any():
            any_errors = True
    if 'classical' in args.calculations:
        global classical_scenarios
        classical_scenarios = make_classical_scenarios()
        print(pd.DataFrame(classical_scenarios).to_string(index=False))
        classical_results = run_sweep('classical', classical_grid(), classical_case)
        print_result_table(classical_results, ['label', 'p_err', 'epsilon', 'n', 'H_bits', 'bits_per_copy', 'solve_seconds'])
        if (classical_results['run_status'] == 'error').any():
            any_errors = True
    return 1 if any_errors else 0

if __name__ == "__main__":
    sys.exit(main())
