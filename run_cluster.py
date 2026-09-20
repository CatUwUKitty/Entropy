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
import numpy as np
import mosek
import mosek.fusion as mf

def validate_cq_blocks(rho_blocks, tol=1e-09):
    """Validate and clean subnormalized CQ blocks."""
    if not rho_blocks:
        raise ValueError('rho_blocks cannot be empty.')
    is_real = all((np.allclose(np.asarray(rho).imag, 0, atol=tol) for rho in rho_blocks))
    dtype = np.float64 if is_real else complex
    blocks = [np.asarray(rho.real if is_real else rho, dtype=dtype) for rho in rho_blocks]
    dE = blocks[0].shape[0]
    for x, rho in enumerate(blocks):
        if rho.shape != (dE, dE):
            raise ValueError(f'Block {x} has inconsistent dimensions: expected ({dE}, {dE}), got {rho.shape}.')
        if is_real:
            if not np.allclose(rho, rho.T, atol=tol):
                raise ValueError(f'Block {x} is not symmetric.')
            blocks[x] = (rho + rho.T) / 2
        else:
            if not np.allclose(rho, rho.conj().T, atol=tol):
                raise ValueError(f'Block {x} is not Hermitian.')
            blocks[x] = (rho + rho.conj().T) / 2
        min_eig = np.min(np.linalg.eigvalsh(blocks[x]))
        if min_eig < -tol:
            raise ValueError(f'Block {x} is not positive semidefinite (min eigenvalue = {min_eig}).')
    total_trace = sum((np.trace(rho).real for rho in blocks))
    if not np.isclose(total_trace, 1.0, atol=tol):
        raise ValueError(f'CQ blocks must have total trace 1, got {total_trace}.')
    return blocks

def check_solver_status(model):
    """Check whether the MOSEK Fusion SDP was solved successfully."""
    status = model.getPrimalSolutionStatus()
    if status != mf.SolutionStatus.Optimal:
        raise RuntimeError(f'MOSEK Fusion did not reach optimality. Status: {status}')
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
import numpy as np
import mosek.fusion as mf
import scipy.sparse as sp

def to_mosek_sparse(matrix: np.ndarray, tol: float=1e-12) -> mf.Matrix:
    """Filter near-zero values and convert to a MOSEK sparse matrix."""
    coo = sp.coo_matrix(matrix)
    mask = np.abs(coo.data) > tol
    return mf.Matrix.sparse(coo.shape[0], coo.shape[1], coo.row[mask].astype(np.int32), coo.col[mask].astype(np.int32), coo.data[mask].astype(np.float64))

def measured_smooth_collision_entropy_fast(n, W=0.85, epsilon=0.001, num_threads=4, verbose=False):
    """
    Exploits the physical bit-flip unitary symmetry U = diag(1, -1, 1, -1).
    Since U_x tau_0 U_x = tau_x, setting B_x = U_x B_0 U_x and C_x = U_x C_0 U_x
    satisfies all constraints identically and collapses 2^n variable pairs (B_x, C_x)
    down to a single pair (B_0, C_0). The coupling sum_x C_x becomes a commutant projection.
    """
    dE = 4 ** n
    nX = 2 ** n
    u_single = np.array([1, -1, 1, -1], dtype=float)
    signs = []
    for k in range(n):
        rep = np.kron(np.ones(4 ** k), np.kron(u_single, np.ones(4 ** (n - 1 - k))))
        signs.append(rep)
    twirl_mask = np.ones((dE, dE), dtype=float)
    for q in range(n):
        s = signs[q]
        twirl_mask *= 1.0 + s[:, None] * s[None, :]
    blocks_1 = generate_werner_cq_blocks(W=W, n=1)
    tau0_single = blocks_1[0].real
    tau0 = tau0_single
    for _ in range(1, n):
        tau0 = np.kron(tau0, tau0_single)
    with mf.Model(f'ms_h2_sym_n{n}') as M:
        M.setSolverParam('numThreads', num_threads)
        M.setSolverParam('intpntSolveForm', 'primal')
        if verbose:
            import sys
            M.setLogHandler(sys.stdout)
        B0 = M.variable('B0', mf.Domain.inPSDCone(dE))
        C0 = M.variable('C0', mf.Domain.inPSDCone(dE))
        t = M.variable('t', mf.Domain.unbounded())
        k = M.variable('k', mf.Domain.unbounded())
        I_d = mf.Matrix.eye(dE)
        I_expr = mf.Expr.constTerm(I_d)
        M.constraint('B0_ub', mf.Expr.sub(mf.Expr.mul(t, I_d), B0), mf.Domain.inPSDCone(dE))
        top = mf.Expr.hstack(C0, B0)
        bot = mf.Expr.hstack(B0, I_expr)
        M.constraint('schur0', mf.Expr.vstack(top, bot), mf.Domain.inPSDCone(2 * dE))
        mask_sparse = to_mosek_sparse(twirl_mask)
        sum_C = mf.Expr.mulElm(mask_sparse, C0)
        M.constraint('sum_C_ub', mf.Expr.sub(mf.Expr.mul(k, I_d), sum_C), mf.Domain.inPSDCone(dE))
        coeff = 2.0 * float(nX)
        tau0_sparse = to_mosek_sparse(tau0)
        tr0 = mf.Expr.dot(B0, tau0_sparse)
        obj = mf.Expr.sub(mf.Expr.sub(mf.Expr.mul(coeff, tr0), mf.Expr.mul(2.0 * epsilon, t)), k)
        M.objective('obj', mf.ObjectiveSense.Maximize, obj)
        M.solve()
        Q_star = float(M.primalObjValue())
        status = str(M.getProblemStatus())
    H2 = -np.log2(Q_star)
    return {'status': status, 'Q_star': Q_star, 'entropy': H2}
for n in range(1, 4):
    res = measured_smooth_collision_entropy_fast(n=n, W=0.85, epsilon=0.001, num_threads=12)
    print(f"H2 ({n} copies): {res['entropy']:.6f} bits | Q*: {res['Q_star']:.8f}")
import numpy as np
import mosek.fusion as mf
import scipy.sparse as sp
num_threads = 4

def to_mosek_sparse(matrix: np.ndarray, tol: float=1e-12) -> mf.Matrix:
    """Filter near-zero values and convert to a MOSEK sparse matrix."""
    coo = sp.coo_matrix(matrix)
    mask = np.abs(coo.data) > tol
    return mf.Matrix.sparse(coo.shape[0], coo.shape[1], coo.row[mask].astype(np.int32), coo.col[mask].astype(np.int32), coo.data[mask].astype(np.float64))

def measured_smooth_collision_entropy(rho_blocks, epsilon=0.001, verbose=False):
    """
    Compute H_2^{epsilon,M,up}(X|E) using Eq. (B8) via MOSEK Fusion.
    rho_blocks[x] = p(x) rho_{E|x}.
    """
    rho_blocks = validate_cq_blocks(rho_blocks)
    nX = len(rho_blocks)
    dE = rho_blocks[0].shape[0]
    if not all((np.allclose(np.asarray(b).imag, 0) for b in rho_blocks)):
        raise NotImplementedError('Density matrix blocks contain non-zero imaginary components. MOSEK Fusion operates on real cones. Real-embedding is required for complex states.')
    rho_blocks = [np.ascontiguousarray(np.real(b), dtype=np.float64) for b in rho_blocks]
    with mf.Model('measured_smooth_h2') as M:
        M.setSolverParam('numThreads', num_threads)
        M.setSolverParam('intpntSolveForm', 'primal')
        if verbose:
            M.setLogHandler(sys.stdout)
        B = [M.variable(f'B_{x}', mf.Domain.inPSDCone(dE)) for x in range(nX)]
        C = [M.variable(f'C_{x}', mf.Domain.inPSDCone(dE)) for x in range(nX)]
        t = M.variable('t', mf.Domain.unbounded())
        k = M.variable('k', mf.Domain.unbounded())
        I_d = mf.Matrix.eye(dE)
        I_expr = mf.Expr.constTerm(I_d)
        for x in range(nX):
            M.constraint(f'B_ub_{x}', mf.Expr.sub(mf.Expr.mul(t, I_d), B[x]), mf.Domain.inPSDCone(dE))
        for x in range(nX):
            top = mf.Expr.hstack(C[x], B[x])
            bot = mf.Expr.hstack(B[x], I_expr)
            M.constraint(f'schur_{x}', mf.Expr.vstack(top, bot), mf.Domain.inPSDCone(2 * dE))
        sum_C = mf.Expr.add(C)
        M.constraint('sum_C_ub', mf.Expr.sub(mf.Expr.mul(k, I_d), sum_C), mf.Domain.inPSDCone(dE))
        rho_sparse = [to_mosek_sparse(b) for b in rho_blocks]
        trace_terms = [mf.Expr.dot(B[x], rho_sparse[x]) for x in range(nX)]
        total_trace = mf.Expr.add(trace_terms)
        obj = mf.Expr.sub(mf.Expr.sub(mf.Expr.mul(2.0, total_trace), mf.Expr.mul(2.0 * epsilon, t)), k)
        M.objective('obj', mf.ObjectiveSense.Maximize, obj)
        M.solve()
        check_solver_status(M)
        Q_star = float(M.primalObjValue())
        if Q_star <= 0:
            raise RuntimeError(f'Expected Q* > 0, got {Q_star}.')
        t_val = float(t.level()[0])
        k_val = float(k.level()[0])
    H2 = -np.log2(Q_star)
    return {'entropy': H2, 'Q_star': Q_star, 't': t_val, 'k': k_val, 'status': 'OPTIMAL'}
for n in range(1, 3):
    blocks = generate_werner_cq_blocks(W=1, n=n)
    res = measured_smooth_collision_entropy(rho_blocks=blocks, epsilon=0.001)
    print(f"H2 ({n} copies): {res['entropy']:.6f} bits | Q*: {res['Q_star']:.8f}")
