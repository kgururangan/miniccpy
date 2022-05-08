# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_cc_calc, run_eomcc_calc, get_hbar

basis = 'dz'
nfrozen = 1

# Define molecule geometry and basis set
geom = [['H', (0, 1.515263, -1.058898)], 
        ['H', (0, -1.515263, -1.058898)], 
        ['O', (0.0, 0.0, -0.0090)]]

T, E0, fock, g, o, v = run_cc_calc(geom, basis, nfrozen, method='ccsdt')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')

nroot = 5
R, omega, r0 = run_eomcc_calc(T, fock, g, H1, H2, o, v, nroot, method='eomccsdt')






