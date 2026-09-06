import os
from pathlib import Path

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
import numpy as np
import cvxpy as cp
import mosek

def validate_cq_blocks(rho_blocks, tol=1e-09):
    """Validate and clean subnormalized CQ blocks."""
    if not rho_blocks:
        raise ValueError('rho_blocks cannot be empty.')
    blocks = [np.asarray(rho, dtype=complex) for rho in rho_blocks]
    dE = blocks[0].shape[0]
    for x, rho in enumerate(blocks):
        if rho.shape != (dE, dE):
            raise ValueError(f'Block {x} has inconsistent dimensions.')
        if not np.allclose(rho, rho.conj().T, atol=tol):
            raise ValueError(f'Block {x} is not Hermitian.')
        blocks[x] = (rho + rho.conj().T) / 2
        if np.min(np.linalg.eigvalsh(blocks[x])) < -tol:
            raise ValueError(f'Block {x} is not positive semidefinite.')
    total_trace = sum((np.trace(rho).real for rho in blocks))
    if not np.isclose(total_trace, 1.0, atol=tol):
        raise ValueError(f'CQ blocks must have total trace 1, got {total_trace}.')
    return blocks

def validate_parameters(epsilon, log_base):
    """Validate scalar parameters."""
    if not 0 <= epsilon < 1:
        raise ValueError('epsilon must satisfy 0 <= epsilon < 1.')
    if log_base <= 0 or np.isclose(log_base, 1):
        raise ValueError('log_base must be positive and not equal to 1.')

def check_solver_status(problem):
    """Check whether the SDP was solved successfully."""
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f'MOSEK returned status: {problem.status}')
    if problem.value is None or not np.isfinite(problem.value):
        raise RuntimeError('MOSEK did not return a finite optimum.')
    if problem.value <= 0:
        raise RuntimeError(f'Expected Q* > 0, got {problem.value}.')

def measured_smooth_collision_entropy(rho_blocks, epsilon=0.0, log_base=2, verbose=False):
    """
    Compute H_2^{epsilon,M,up}(X|E) using Eq. (B8).
    rho_blocks[x] = p(x) rho_{E|x}.
    """
    validate_parameters(epsilon, log_base)
    rho_blocks = validate_cq_blocks(rho_blocks)
    nX = len(rho_blocks)
    dE = rho_blocks[0].shape[0]
    I = np.eye(dE)
    B = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    C = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    t = cp.Variable()
    k = cp.Variable()
    constraints = []
    for x in range(nX):
        constraints += [B[x] >> 0, C[x] >> 0, B[x] << t * I, cp.bmat([[C[x], B[x]], [B[x], I]]) >> 0]
    constraints += [sum(C) << k * I]
    trace_term = sum((cp.real(cp.trace(B[x] @ rho_blocks[x])) for x in range(nX)))
    objective = cp.Maximize(2 * trace_term - 2 * epsilon * t - k)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND)
    check_solver_status(problem)
    Q_star = float(problem.value)
    H2 = -np.log(Q_star) / np.log(log_base)
    return {'entropy': H2, 'Q_star': Q_star, 'B': [Bx.value for Bx in B], 'C': [Cx.value for Cx in C], 't': t.value, 'k': k.value, 'status': problem.status}
zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)
plus = (zero + one) / np.sqrt(2)
proj_0 = np.outer(zero, zero.conj())
proj_1 = np.outer(one, one.conj())
proj_plus = np.outer(plus, plus.conj())
identity = np.eye(2, dtype=complex)
test_states = {'Perfect Correlation (H2 = 0)': [0.5 * proj_0, 0.5 * proj_1], 'Uniform Independent (|0><0|) (H2 = 1)': [0.5 * proj_0, 0.5 * proj_0], 'Uniform Independent (Mixed I/2) (H2 = 1)': [0.25 * identity, 0.25 * identity], 'BB84 Overlap (|0> and |+>) (H2 ≈ 0.2284)': [0.5 * proj_0, 0.5 * proj_plus]}
epsilon = 1e-10
header = f"{'State Name':<42} | {'Status':<10} | {'Q*':<14} | {'H2 (bits)':<12}"
print(header)
print('-' * len(header))
for name, blocks in test_states.items():
    res = measured_smooth_collision_entropy(rho_blocks=blocks, epsilon=epsilon, log_base=2)
    print(f"{name:<42} | {res['status']:<10} | {res['Q_star']:<14.10f} | {res['entropy']:<12.6f}")
