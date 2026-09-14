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
import itertools
import numpy as np
import mosek.fusion as mf
import scipy.sparse as sp

def to_mosek_sparse(matrix: np.ndarray, tol: float=1e-12) -> mf.Matrix:
    """Filter near-zero values and convert to a MOSEK sparse matrix."""
    coo = sp.coo_matrix(matrix)
    mask = np.abs(coo.data) > tol
    return mf.Matrix.sparse(coo.shape[0], coo.shape[1], coo.row[mask].astype(np.int32), coo.col[mask].astype(np.int32), coo.data[mask].astype(np.float64))

def measured_smooth_collision_entropy_block_diag(n, W=0.85, epsilon=0.001, num_threads=4, verbose=False):
    """
    Solves the Measured Smooth Conditional Collision Entropy SDP by direct-sum block diagonalization.

    Exploits the exact block structure of tau_0 and the bit-flip twirling operator:
    - Splits the (2 * 4^n) Schur cone into 2^n independent cones of size (2 * 2^n).
    - Preserves high-precision MOSEK convergence while eliminating the O(256^n) memory wall.
    """
    d_sub = 2 ** n
    num_blocks = 2 ** n
    nX = 2 ** n
    lam0 = (1.0 + 3.0 * W) / 4.0
    lam1 = (1.0 - W) / 4.0
    tau0_0 = 0.5 * np.array([[lam0, np.sqrt(lam0 * lam1)], [np.sqrt(lam0 * lam1), lam1]], dtype=float)
    tau0_1 = 0.5 * np.array([[lam1, lam1], [lam1, lam1]], dtype=float)
    tau_base = [tau0_0, tau0_1]
    tau0_blocks = []
    for bits in itertools.product([0, 1], repeat=n):
        blk = tau_base[bits[0]]
        for bit in bits[1:]:
            blk = np.kron(blk, tau_base[bit])
        tau0_blocks.append(blk)
    u_sub = np.array([1.0, -1.0])
    twirl_mask_sub = np.ones((d_sub, d_sub), dtype=float)
    for q in range(n):
        s = np.kron(np.ones(2 ** q), np.kron(u_sub, np.ones(2 ** (n - 1 - q))))
        twirl_mask_sub *= 1.0 + s[:, None] * s[None, :]
    mask_sparse = to_mosek_sparse(twirl_mask_sub)
    I_sub = mf.Matrix.eye(d_sub)
    I_expr_sub = mf.Expr.constTerm(I_sub)
    with mf.Model(f'ms_h2_block_diag_n{n}') as M:
        M.setSolverParam('numThreads', num_threads)
        M.setSolverParam('intpntSolveForm', 'primal')
        if verbose:
            import sys
            M.setLogHandler(sys.stdout)
        t = M.variable('t', mf.Domain.unbounded())
        k = M.variable('k', mf.Domain.unbounded())
        B_blocks = []
        C_blocks = []
        tr_terms = []
        for b in range(num_blocks):
            Bb = M.variable(f'B_{b}', mf.Domain.inPSDCone(d_sub))
            Cb = M.variable(f'C_{b}', mf.Domain.inPSDCone(d_sub))
            B_blocks.append(Bb)
            C_blocks.append(Cb)
            M.constraint(f'B_ub_{b}', mf.Expr.sub(mf.Expr.mul(t, I_sub), Bb), mf.Domain.inPSDCone(d_sub))
            top = mf.Expr.hstack(Cb, Bb)
            bot = mf.Expr.hstack(Bb, I_expr_sub)
            M.constraint(f'schur_{b}', mf.Expr.vstack(top, bot), mf.Domain.inPSDCone(2 * d_sub))
            sum_Cb = mf.Expr.mulElm(mask_sparse, Cb)
            M.constraint(f'sum_C_ub_{b}', mf.Expr.sub(mf.Expr.mul(k, I_sub), sum_Cb), mf.Domain.inPSDCone(d_sub))
            tau_sparse = to_mosek_sparse(tau0_blocks[b])
            tr_terms.append(mf.Expr.dot(Bb, tau_sparse))
        coeff = 2.0 * float(nX)
        total_tr = mf.Expr.add(tr_terms)
        obj = mf.Expr.sub(mf.Expr.sub(mf.Expr.mul(coeff, total_tr), mf.Expr.mul(2.0 * epsilon, t)), k)
        M.objective('obj', mf.ObjectiveSense.Maximize, obj)
        M.solve()
        Q_star = float(M.primalObjValue())
        status = str(M.getProblemStatus())
    H2 = -np.log2(Q_star)
    return {'status': status, 'Q_star': Q_star, 'entropy': H2}
W = 0.85
max_Copies = 6
epsilon = 0.001
for n in range(1, max_Copies + 1):
    res = measured_smooth_collision_entropy_block_diag(n=n, W=W, epsilon=epsilon, num_threads=12)
    print(f"H2, W = {W} ({n} copies): {res['entropy']:.6f} bits | Q*: {res['Q_star']:.8f}")
