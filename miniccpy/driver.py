import time
from importlib import import_module

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))

__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
MODULES = [module for module in __all__]

def run_cc_calc(geometry, basis, nfrozen, method):

    from pyscf import gto, scf

    from miniccpy.printing import print_system_information
    from miniccpy.integrals import get_integrals_from_pyscf, get_fock
    from miniccpy.energy import hf_energy, cc_energy

    mol = gto.Mole()

    mol.build(
        atom=geometry,
        basis=basis,
        charge=0,
        spin=0,
        cart=False,
        unit='Bohr',
        symmetry=True,
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    # 1-, 2-electron spinorbital integrals in physics notation
    e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_pyscf(mf)

    corr_occ = slice(2 * nfrozen, mf.mol.nelectron)
    corr_unocc = slice(mf.mol.nelectron, 2 * mf.mo_coeff.shape[1])

    print_system_information(mf, nfrozen, e_hf)

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    cc_mod = import_module("miniccpy."+method.lower())
    cc_calculation = getattr(cc_mod, 'kernel')

    tic = time.time()
    T, e_corr = cc_calculation(fock, e2int, corr_occ, corr_unocc)
    toc = time.time()

    minutes, seconds = divmod(toc - tic, 60)

    print("")
    print("    CC Correlation Energy: {: 20.12f}".format(e_corr))
    print("    CC Total Energy:       {: 20.12f}".format(e_corr + e_hf))
    print("")
    print("CC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))

    return T, e_corr+e_hf, fock, e2int, corr_occ, corr_unocc

def get_hbar(T, fock, g, o, v, method):

    # import the specific CC method module and get its update function
    mod = import_module("miniccpy.hbar")
    hbar_builder = getattr(mod, 'build_hbar_'+method.lower())

    H1, H2 = hbar_builder(T, fock, g, o, v)

    return H1, H2

def run_eomcc_calc(T, fock, g, H1, H2, o, v, nroot, method):

    from miniccpy.initial_guess import get_initial_guess

    # check if requested EOMCC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # get the initial guess
    R0, omega0 = get_initial_guess(fock, g, o, v, nroot) 

    R = [0 for i in range(nroot)]
    omega = [0 for i in range(nroot)]
    r0 = [0 for i in range(nroot)]
    for n in range(nroot):
        tic = time.time()
        R[n], omega[n], r0[n] = calculation(R0[:, n], T, omega0[n], H1, H2, o, v)
        toc = time.time()

        minutes, seconds = divmod(toc - tic, 60)

        print("")
        print("    EOMCC Excitation Energy: {: 20.12f}".format(omega[n]))
        print("")
        print("EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))

    return R, omega, r0



