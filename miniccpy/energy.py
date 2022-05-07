import numpy as np

def cc_energy(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0> :


    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """

    #	  1.0000 f(i,a)*t1(a,i)
    energy = 1.0 * np.einsum('ia,ai->', f[o, v], t1)

    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * np.einsum('ijab,abij->', g[o, o, v, v], t2)

    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += 0.5 * np.einsum('ijab,ai,bj->', g[o, o, v, v], t1, t1)

    return energy

def hf_energy(z, g, o):

    energy = np.einsum('ii->', z[o, o])
    energy += 0.5 * np.einsum('ijij->', g[o, o, o, o])

    return energy

def hf_energy_from_fock(f, g, o):

    energy = np.einsum('ii->', f[o, o])
    energy -= 0.5 * np.einsum('ijij->', g[o, o, o, o])

    return energy
