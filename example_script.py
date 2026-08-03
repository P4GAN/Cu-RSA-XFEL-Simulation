import os
notebook_path = os.path.abspath("__file__")
notebook_directory = os.path.dirname(notebook_path)
XLO_path = notebook_directory + '/XLO_sim/'
mini_ocelot_path = notebook_directory + '/mini_ocelot/'

import sys
sys.path.append(XLO_path)
sys.path.append(mini_ocelot_path)

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from XLO_sim import *
import Plot as XLO_plot


from numpy import random
from numpy.linalg import norm
from math import factorial
from numpy import inf, complex128, complex64
import scipy
import numpy.fft as fft
from copy import deepcopy


from o_globals import *
from math_op import find_nearest_idx, fwhm, std_moment, bin_scale, bin_array, mut_coh_func
from py_func import filename_from_path
from ocelog import *
from new_wave import *
import scipy.constants as sp_const
au_in_eV = sp_const.value('atomic unit of energy') / sp_const.value('atomic unit of charge')

import tools


X = XLO_sim("/Users/parkinpham/Programming/Physics/DESY_Internship/XLO_sim_related_notebook/config/Cu-seed_40.00uJ.yaml")

seed_field = tools.Gaussian_pulse_aniso_seed(X)
X.configure(seed_field)
X.run_3D()