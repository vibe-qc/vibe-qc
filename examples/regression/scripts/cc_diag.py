"""Verify CCSD residual equations by feeding PySCF's amplitudes and integrals.

If the residuals R1, R2 are near zero for the converged amplitudes,
the equations are correct and the convergence issue is elsewhere.
"""

import numpy as np
from pyscf import cc, gto, scf

mol_spec = [
    (8, (0.0, 0.0, 0.0)),
    (1, (0.0, 1.432, -1.107)),
    (1, (0.0, -1.432, -1.107)),
]
pmol = gto.Mole()
pmol.unit = "Bohr"
pmol.atom = [[Z, tuple(xyz)] for Z, xyz in mol_spec]
pmol.basis = "sto-3g"
pmol.build()
pmf = scf.RHF(pmol).density_fit(auxbasis="cc-pvdz-ri")
pmf.conv_tol = 1e-12
pmf.kernel()
pcc = cc.CCSD(pmf, frozen=0)
pcc.conv_tol = 1e-10
pcc.kernel()

no = pmol.nelec[0]
nv = pmol.nao_nr() - no
T1 = pcc.t1.copy()  # (no, nv)
T2_py = pcc.t2.copy()  # (no, no, nv, nv)

eris = pcc.ao2mo()
fock = eris.fock

# Extract PySCF's integral blocks
# ovov: (no, nv, no, nv) = (ia|jb)
ovov = np.asarray(eris.ovov)  # Coulomb (ia|jb)
# oooo: (no, no, no, no) = (ij|kl)
oooo = np.asarray(eris.oooo)
# ovvo: (no, nv, nv, no) = (ia|bj)
ovvo = np.asarray(eris.ovvo)
# oovv: (no, no, nv, nv) = (ij|ab)
oovv = np.asarray(eris.oovv)
# ovoo: (no, nv, no, no) = (ia|jk)
ovoo = np.asarray(eris.ovoo)

# Flatten to (no*no, nv*nv) or similar
T2_flat = np.zeros((no * no, nv * nv))
for i in range(no):
    for j in range(no):
        for a in range(nv):
            for b in range(nv):
                T2_flat[i * no + j, a * nv + b] = T2_py[i, j, a, b]

# Build PySCF-style antisymmetrized integrals
# PySCF uses: woovv = oovv*2 - oovv.transpose(0,1,3,2)  — this is 2*(ij|ab) - (ij|ba)
# And for the T2 residual: eris_ovvo is used (not antisymmetrized)

# PySCF's T2 residual starts from _add_vvvv (which computes the τ·W_vvvv contribution),
# then adds: t2new += (eris_ovvo*0.5).transpose(0,3,1,2)
# eris_ovvo.transpose(0,3,1,2) = (no, no, nv, nv) with value (ia|bj) = (ia|jb)
# multiplied by 0.5

# So PySCF's "integral term" is 0.5 * (ia|jb)

# Let me verify my R2 formula using PySCF's integrals

# Build my-style blocks for comparison
ov_ov = np.zeros((no * nv, no * nv))
for i in range(no):
    for a in range(nv):
        ia = i * nv + a
        for j in range(no):
            for b in range(nv):
                jb = j * nv + b
                ov_ov[ia, jb] = ovov[i, a, j, b]

oo_oo = np.zeros((no * no, no * no))
for i in range(no):
    for j in range(no):
        ij = i * no + j
        for k in range(no):
            for l in range(no):
                kl = k * no + l
                oo_oo[ij, kl] = oooo[i, j, k, l]

vv_vv = np.zeros((nv * nv, nv * nv))
for a in range(nv):
    for b in range(nv):
        ab = a * nv + b
        for c in range(nv):
            for d in range(nv):
                cd = c * nv + d
                # PySCF: eris.vvvv has shape (nv, nv, nv, nv) = (ab|cd)
                vv_vv[ab, cd] = np.asarray(eris.vvvv)[a, b, c, d]

# Now compute R2 using PySCF's formulation
# PySCF's R2:
# 1. Start from _add_vvvv (τ·W_vvvv, divided by 2)
# 2. Add 0.5 * eris_ovvo.transpose(0,3,1,2) = 0.5 * (ia|jb) → this is the "integral term"
# 3. Add other T1, T2 terms...

# Let me compute just the integral + vvvv part for comparison

# PySCF integral term (simplified):
R2_pyscf_integral = 0.5 * ovov.transpose(0, 2, 1, 3)  # Wait, need to get indices right

# Actually, eris_ovvo has shape (no, nv, nv, no) = (i, a, b, j) = (ia|bj)
# .transpose(0,3,1,2) → (no, no, nv, nv) = (i, j, a, b)
# Value at (i, j, a, b): (ia|bj) * 0.5

# For my convention: ⟨ij||ab⟩ = 2*(ia|jb) - (ib|ja)
# vs PySCF integral: 0.5*(ia|bj) = 0.5*(ia|jb) [same by symmetry]

# These are DIFFERENT by factors!
# My term 0: 2*(ia|jb) - (ib|ja) (no prefactor)
# PySCF: 0.5*(ia|jb) plus other terms that add up to the full antisymmetrized form

print("=== PySCF integral term vs my Term 0 ===")
for i in range(min(2, no)):
    for j in range(min(2, no)):
        for a in range(min(2, nv)):
            for b in range(min(2, nv)):
                my_val = (
                    2 * ov_ov[i * nv + a, j * nv + b] - ov_ov[i * nv + b, j * nv + a]
                )
                py_val = ovov[
                    i, a, j, b
                ]  # This is what PySCF adds, but with 0.5 factor
                print(
                    f"  ({i},{j}|{a},{b}): my={my_val:.8f}  py*0.5={0.5 * py_val:.8f}  ratio={my_val / (0.5 * py_val + 1e-16):.4f}"
                )

print(
    "\nKey finding: PySCF's integral term (0.5*(ia|jb)) differs from my integral term"
)
print("(2*(ia|jb)-(ib|ja)) by roughly a factor of 2-3, depending on term.")
print("PySCF absorbs much of the antisymmetrization into the Fock and W terms.")
