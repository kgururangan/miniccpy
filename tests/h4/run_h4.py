# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_cc_calc

basis = 'dz'
nfrozen = 0

geom = [['H', (-1.000, -1.000/2, 0.000)], 
        ['H', (-1.000,  1.000/2, 0.000)], 
        ['H', ( 1.000, -1.000/2, 0.000)], 
        ['H', ( 1.000,  1.000/2, 0.000)]]


run_cc_calc(geom, basis, nfrozen, method='ccsd')







