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


import numpy as np
import cvxpy as cp

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

def check_solver_status(problem):
    """Check whether the SDP was solved successfully."""
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f'Solver returned status: {problem.status}')
    if problem.value is None or not np.isfinite(problem.value):
        raise RuntimeError('Solver did not return a finite optimum.')
    if problem.value <= 0:
        raise RuntimeError(f'Expected Q* > 0, got {problem.value}.')

def measured_smooth_collision_entropy(rho_blocks, epsilon=0.001, verbose=False):
    """
    Compute H_2^{epsilon,M,up}(X|E) using Eq. (B8).
    rho_blocks[x] = p(x) rho_{E|x}.
    """
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
        constraints += [B[x] >> 0, B[x] << t * I, cp.bmat([[C[x], B[x]], [B[x], I]]) >> 0]
    constraints += [sum(C) << k * I]
    trace_term = sum((cp.real(cp.trace(B[x] @ rho_blocks[x])) for x in range(nX)))
    objective = cp.Maximize(2 * trace_term - 2 * epsilon * t - k)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND)
    check_solver_status(problem)
    Q_star = float(problem.value)
    H2 = -np.log2(Q_star)
    return {'entropy': H2, 'Q_star': Q_star, 'B': [Bx.value for Bx in B], 'C': [Cx.value for Cx in C], 't': t.value, 'k': k.value, 'status': problem.status}
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
W = 0.85
num_Copies = 2
for i in range(1, num_Copies + 1):
    blocks = generate_werner_cq_blocks(W=0.85, n=num_Copies)
    res = measured_smooth_collision_entropy(rho_blocks=blocks, epsilon=0.001)
    print(f"H2 ({i} copies): {res['entropy']:.6f} bits | Q*: {res['Q_star']:.8f}")
