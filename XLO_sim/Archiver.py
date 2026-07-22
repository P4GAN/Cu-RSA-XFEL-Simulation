import yaml, sys, h5py, time
import numpy as np
from XLO_sim import *
from numpy.random import randint
from datetime import datetime

class XLO_Archiver:

    def __init__(self, sim_config, archive_config, data_file):
        """
        Initiates the XLO_Archiver object. This object holds the data of interest for multiple realizations (run) of XLO_sim. The extent of this data is specified by the parameters in the run configuration file.
        
        The object can hold either the data for the whole numerical grid (save_full_data: True), 3D arrays for specified indices on an axis (in the following named cubes; save_data_cubes: True), or cuts of data specified by an array of indices (indexed data; save_indexed_data: True). 
        
        In the configuration file, data cubes are specified by the parameter "cubes", which is a dictionary with keys being the names of the cubes along different axes ("icube", where i = t,x,y,z) and values arrays of indices for which the data is saved over a given axis. For example: if "tcube: [1]", the saved data for variable rho is rho[1, :, :, :], where the order of axes is t,x,y,z; if "zcube: [0, -1]", the saved data is np.array([rho[:, :, :, 0], rho[:, :, :, -1]]). Values of "icubes" are lists of integers; if a key is omitted of if an empty list is passed, no data cube over this axis is saved.

        Indexed data is specified by the dictionary "indices", the keys of which are the names of the cuts (to be used for saving data). The value for each key is a list with 4 elements specifying the [t,x,y,z] indices for which the data is saved. Possible elements of the list are:
        - "all": all indices over this dimension are taken (in numpy notation equivalent to ":")
        - "mid": the middle index over an axis; if nmax is the number of grid points over an axis, mid = int(nmax/2)
        - integer: explicitly specified index.
        Foe example: if the value of "cut1" is [-1, mid, mid, all], the saved data for variable "rho" is rho[-1, int(xgrid/2), int(ygrid/2), :].

        The object can store only the data averaged over the different realizations (save_averages: True) and / or data for the different realizations (save_individual_runs: True). It always stores the number of photons for the fields, optionally it can hold also the field fluxes and amplitude of the ASE field (save_fields: True), and the field populations (save_populations: True) or all density matrix elements (save_density_matrix: True).

        Data can be saved into an HDF5 file, the name of which is specified by data_file. Either a new file is created (data_save_flag: new) or data is appended to an existing file (data_save_flag: append). Data is only appended if all the simulation and run parameters for the new run, except for the number of repetitions and processes, are the same as in the existing file. In this case, average arrays are updated with the newly calculated data, and data for the new realizations of the simulation is appended to the existing datasets in the file. If the simulation and run parameters do not match, data is saved into a new file, the name of which contains the name specified in data_file.

        Parameters
        ----------
        sim_config : str
            Name of the configuration file for XLO_sim.
        
        archive_config : str
            Name of the run configuration file.
            
        data_file: str
            Name of the h5 file.


        Returns
        -------

        """

        self.config = yaml.load(open(archive_config), Loader=yaml.FullLoader)
        self.sim_config_name = sim_config
        self.data_file = data_file

        save_flags = ["save_averages", "save_individual_runs", "save_full_grid", "save_data_cubes", "save_indexed_data"]
        for flag in save_flags:
            if flag not in self.config.keys():
                self.config[flag] = False

        variable_flags = ["save_fields", "save_populations", "save_density_matrix", "save_gain_polarization", "save_combined_populations"]
        for flag in variable_flags:
            if flag not in self.config.keys():
                self.config[flag] = False

        for k, v in self.config.items():
            setattr(self, k, v)

        if ("data_save_flag" not in self.config.keys()) or (self.data_save_flag == 'append'):
            self.h5_flag = 'a'
        elif self.data_save_flag == 'new':
            self.h5_flag = 'w'
        else:
            print("Invalid data_save_flag, setting to 'append'")
            self.h5_flag = 'a'

        self.XLO_sim_reference = XLO_sim(sim_config)
        if self.XLO_sim_reference.random_seed == None:
            self.XLO_sim_reference.random_seed = -1

        for sim_pars in ["tgrid", "xgrid", "ygrid", "zgrid", "nlevel"]:
            setattr(self, sim_pars, getattr(self.XLO_sim_reference, sim_pars))

        self.sim_attributes = ["tgrid", "xgrid", "ygrid", "zgrid", "tmax", "xmax", "ymax", "zmax", "pump_pulse_format", "sigma_t", "N_pump_photons", "zR", "N_modes", "sigma_coh", "M2", "is_spontaneous_only", "random_seed", "run_mode", "enable_pump_diffraction", "enable_self_absorption"]


    def make_arrays_for_data(self, data_name, grid_dims, is_average=False):
        """
        Creates the arrays for storing all variables of interest for the full grid, a data cube or a cut of indexed data. Either the arrays for average data or for all realizations are created.

        Parameters
        ----------
        data_name : string
            The suffix for the array names specifying the type of the arrays.

        grid_dims : tuple
            Dimensions of the arrays over the temporal and spatial axes (t,x,y,z).

        is_average : bool
            If True, the arrays for storing average data are created, otherwise the arrays for storing data for different realizations.

        Returns
        -------

        """
        if is_average:
            rep_size = ()
            name_is_avg = "avg"
        else:
            rep_size = (self.nrep,)
            name_is_avg = "reps"

        name_append = "_"+data_name+"_"+name_is_avg

        if self.save_fields:
            setattr(self, "j_3D"+name_append, np.zeros(rep_size+grid_dims))
            setattr(self, "Omega_pstxyz"+name_append, np.zeros(rep_size+(2, 2)+grid_dims, dtype=complex))
            setattr(self, "J_Omega_stxyz"+name_append, np.zeros(rep_size+(2,)+grid_dims))

        if self.save_populations or self.save_density_matrix:
            setattr(self, "rho_0_3D"+name_append, np.zeros(rep_size+grid_dims))
            
        if self.save_density_matrix:
            setattr(self, "rho_ijtxyz"+name_append, np.zeros(rep_size+(self.nlevel, self.nlevel)+grid_dims, dtype=complex))

        elif self.save_populations:
            setattr(self, "rho_itxyz"+name_append, np.zeros(rep_size+(self.nlevel,)+grid_dims, dtype=complex))

        if self.save_gain_polarization:
            setattr(self, "P_pstxyz"+name_append, np.zeros(rep_size+(2, 2)+grid_dims, dtype=complex))

        if self.save_combined_populations:
            for name in ["E_sstxyz", "G_sstxyz"]:
                setattr(self, name+name_append, np.zeros(rep_size+(2, 2)+grid_dims, dtype=complex))

    def make_all_arrays_for_data(self, data_name, grid_dims):
        """
        Creates the both average arrays and arrays for different realizations for storing all variables of interest for the full grid, a data cube or a cut of indexed data.

        Parameters
        ----------
        data_name : string
            The suffix for the array names specifying the type of the arrays.

        grid_dims : tuple
            Dimensions of the arrays over the temporal and spatial axes (t,x,y,z).

        Returns
        -------

        """
        if self.save_averages:
            self.make_arrays_for_data(data_name, grid_dims, is_average=True)
        if self.save_individual_runs:
            self.make_arrays_for_data(data_name, grid_dims)

    def generate_dimensions_for_indexed_data(self, cut_name):
        """
        Generates (t,x,y,z) tuple of array dimensions for a given cut of indexed data.

        Parameters
        ----------
        cut_name : string
            Name of the cut of indexed data.

        Returns
        -------
        out: tuple
            Dimensions of the arrays over the temporal and spatial axes (t,x,y,z).

        """
        index_array = self.indices[cut_name]
        full_grid = (self.tgrid, self.xgrid, self.ygrid, self.zgrid)
        out = ()
        for i in range(4):
            if index_array[i] == 'all':
                out += (full_grid[i],)
            else:
                continue
        return out

    def generate_dimensions_for_cubes(self, cube_name):
        """
        Generates (t,x,y,z) tuple of array dimensions for a given data cube.

        Parameters
        ----------
        cube_name : string
            Name of the data cube.

        Returns
        -------
        out: tuple
            Dimensions of the arrays over the temporal and spatial axes (t,x,y,z).

        """
        names = ["tcube", "xcube", "ycube", "zcube"]
        grid_dims = [self.tgrid, self.xgrid, self.ygrid, self.zgrid]
        cube_index = names.index(cube_name)
        grid_dims[cube_index] = len(self.cubes[cube_name])
        return tuple(grid_dims)

    def generate_np_indices(self, cut_name, prefix=None):
        """
        Generates a numpy index tuple of array dimensions for the full data or a given cut of indexed data.

        Parameters
        ----------
        cut_name : string
            Name of the cut of indexed data.

        prefix: None or integer
            Number of additional axes (inserted at the beginning) over which all indices are taken. 

        Returns
        -------
        numpy Index Expression
            Numpy array indices.

        """
        if cut_name == "full":
            return np.s_[:]

        index_array = self.indices[cut_name]
        full_grid = (self.tgrid, self.xgrid, self.ygrid, self.zgrid)
        idx_string = ""
        if prefix is not None:
            for i in range(prefix):
                idx_string += ":, "
        for i in range(4):
            if index_array[i] == 'all':
                idx_string += ":"
            elif index_array[i] == 'mid':
                idx_string += f"{int(full_grid[i]/2)}"
            else:
                idx_string += f"{index_array[i]}"
            
            if i != 3:
                idx_string += ", "
        return eval(f"np.s_[{idx_string}]")

    def setup_arrays(self):
        """
        Adds all the arrays for storing the data of interest to the XLO_Archiver object.

        Parameters
        ----------

        Returns
        -------

        """

        if self.save_averages:
            self.nphoton_pump_z_avg = np.zeros(self.zgrid)
            self.nphoton_sz_avg = np.zeros((2, self.zgrid), dtype=complex)
            self.nphoton_reg_sz_avg = np.zeros((2, self.zgrid), dtype=complex)
        if self.save_individual_runs:
            self.nphoton_pump_z_reps = np.zeros((self.nrep, self.zgrid))
            self.nphoton_sz_reps = np.zeros((self.nrep, 2, self.zgrid), dtype=complex)
            self.nphoton_reg_sz_reps = np.zeros((self.nrep, 2, self.zgrid), dtype=complex)

        if self.save_full_grid:
            grid_dims = (self.tgrid, self.xgrid, self.ygrid, self.zgrid)
            self.make_all_arrays_for_data("full", grid_dims)

        elif self.save_data_cubes:
            for cube_name in self.cubes.keys():
                if self.cubes[cube_name]:
                    grid_dims = self.generate_dimensions_for_cubes(cube_name)
                    self.make_all_arrays_for_data(cube_name, grid_dims)

        elif self.save_indexed_data:
            for cut_name in self.indices.keys():
                grid_dims = self.generate_dimensions_for_indexed_data(cut_name)
                self.make_all_arrays_for_data(cut_name, grid_dims)

    def update_arrays_for_cube(self, X, cube_name, axis_idx, rep_idx):
        """
        Updates arrays for different realizations for data cubes over a specified axis.

        Parameters
        ----------
        X : XLO_sim
            XLO_sim object for one realization of the simulation.

        cube_name : str
            String indicating on which axis the cubes are taken. Allowed values are "tcube", "xcube", "ycube" or "zcube".

        axis_idx : integer
            Index of the axis.

        rep_idx : int
            Index of the XLO_sim realization.

        Returns
        -------

        """

        name_append = "_"+cube_name+"_reps"
        cubes = self.cubes[cube_name][:]

        if self.save_fields:
            getattr(self, "j_3D"+name_append)[rep_idx] = np.take(X.j_3D, cubes, axis=axis_idx)
            getattr(self, "Omega_pstxyz"+name_append)[rep_idx] = np.take(X.Omega_pstxyz, cubes, axis=axis_idx+2)
            getattr(self, "J_Omega_stxyz"+name_append)[rep_idx] = np.real(np.take(X.Omega_pstxyz[0], cubes, axis=axis_idx+1) * np.take(X.Omega_pstxyz[1], cubes, axis=axis_idx+1))

        if self.save_populations or self.save_density_matrix:
            getattr(self, "rho_0_3D"+name_append)[rep_idx] = np.take(X.rho_0_3D, cubes, axis=axis_idx)
        
        if self.save_density_matrix:
            getattr(self, "rho_ijtxyz"+name_append)[rep_idx] = np.take(X.rho_ijtxyz, cubes, axis=axis_idx)

        elif self.save_populations:
            getattr(self, "rho_itxyz"+name_append)[rep_idx] = np.take(np.moveaxis(np.diagonal(X.rho_ijtxyz, axis1=0, axis2=1), -1, 0), cubes, axis=axis_idx+1)

        if self.save_gain_polarization:
            getattr(self, "P_pstxyz"+name_append)[rep_idx] = np.take(X.P_pstxyz, cubes, axis=axis_idx+2)

        if self.save_combined_populations:
            getattr(self, "E_sstxyz"+name_append)[rep_idx] = np.take(X.E_sstxyz, cubes, axis=axis_idx+2)
            getattr(self, "G_sstxyz"+name_append)[rep_idx] = np.take(X.G_sstxyz, cubes, axis=axis_idx+2)

    def update_arrays_for_full_or_cut(self, X, cut_name, rep_idx):
        """
        Updates arrays for different realizations for the full data or indexed data for cuts.

        Parameters
        ----------
        X : XLO_sim
            XLO_sim object for one realization of the simulation.
        
        cut_name : string
            Name of the cut of indexed data or "full".

        rep_idx : int
            Index of the XLO_sim realization.

        Returns
        -------

        """

        name_append = "_"+cut_name+"_reps"
        slice_ = self.generate_np_indices(cut_name)
        slice1_ = self.generate_np_indices(cut_name, prefix=1)
        slice2_ = self.generate_np_indices(cut_name, prefix=2)

        if self.save_fields:
            getattr(self, "j_3D"+name_append)[rep_idx] = X.j_3D[slice_]
            getattr(self, "Omega_pstxyz"+name_append)[rep_idx] = X.Omega_pstxyz[slice2_]
            getattr(self, "J_Omega_stxyz"+name_append)[rep_idx] = np.real(X.Omega_pstxyz[0][slice1_] * X.Omega_pstxyz[1][slice1_])

        if self.save_populations or self.save_density_matrix:
            getattr(self, "rho_0_3D"+name_append)[rep_idx] = X.rho_0_3D[slice_]
        
        if self.save_density_matrix:
            getattr(self, "rho_ijtxyz"+name_append)[rep_idx] = X.rho_ijtxyz[slice_]

        elif self.save_populations:
            getattr(self, "rho_itxyz"+name_append)[rep_idx] = np.moveaxis(np.diagonal(X.rho_ijtxyz, axis1=0, axis2=1), -1, 0)[slice1_]

        if self.save_gain_polarization:
            getattr(self, "P_pstxyz"+name_append)[rep_idx] = X.P_pstxyz[slice2_]

        if self.save_combined_populations:
            getattr(self, "E_sstxyz"+name_append)[rep_idx] = X.E_sstxyz[slice2_]
            getattr(self, "G_sstxyz"+name_append)[rep_idx] = X.G_sstxyz[slice2_]

    def update_average_for_cube(self, X, cube_name, axis_idx):
        """
        Updates average arrays for data cubes over a specified axis.

        Parameters
        ----------
        X : XLO_sim
            XLO_sim object for one realization of the simulation.

        cube_name : str
            String indicating on which axis the cubes are taken. Allowed values are "tcube", "xcube", "ycube" or "zcube".

        axis_idx : integer
            Index of the axis.

        Returns
        -------

        """
        name_append = "_"+cube_name+"_avg"
        cubes = self.cubes[cube_name]

        if self.save_fields:
            getattr(self, "j_3D"+name_append)[:] += np.take(X.j_3D, cubes, axis=axis_idx) / self.nrep
            getattr(self, "Omega_pstxyz"+name_append)[:] += np.take(X.Omega_pstxyz, cubes, axis=axis_idx+2) / self.nrep
            getattr(self, "J_Omega_stxyz"+name_append)[:] += np.real(np.take(X.Omega_pstxyz[0], cubes, axis=axis_idx+1) * np.take(X.Omega_pstxyz[1], cubes, axis=axis_idx+1)) / self.nrep

        if self.save_populations or self.save_density_matrix:
            getattr(self, "rho_0_3D"+name_append)[:] += np.take(X.rho_0_3D, cubes, axis=axis_idx) / self.nrep
        
        if self.save_density_matrix:
            getattr(self, "rho_ijtxyz"+name_append)[:] += np.take(X.rho_ijtxyz, cubes, axis=axis_idx) / self.nrep

        elif self.save_populations:
            getattr(self, "rho_itxyz"+name_append)[:] += np.take(np.moveaxis(np.diagonal(X.rho_ijtxyz, axis1=0, axis2=1), -1, 0), cubes, axis=axis_idx+1) / self.nrep

        if self.save_gain_polarization:
            getattr(self, "P_pstxyz"+name_append)[:] += np.take(X.P_pstxyz, cubes, axis=axis_idx+2) / self.nrep

        if self.save_combined_populations:
            getattr(self, "E_sstxyz"+name_append)[:] += np.take(X.E_sstxyz, cubes, axis=axis_idx+2) / self.nrep
            getattr(self, "G_sstxyz"+name_append)[:] += np.take(X.G_sstxyz, cubes, axis=axis_idx+2) / self.nrep

    def update_average_for_full_or_cut(self, X, cut_name):
        """
        Updates average arrays for the full data or indexed data for cuts.

        Parameters
        ----------
        X : XLO_sim
            XLO_sim object for one realization of the simulation.

        cut_name : string
            Name of the cut of indexed data or "full".

        Returns
        -------

        """
        name_append = "_"+cut_name+"_avg"
        slice_ = self.generate_np_indices(cut_name)
        slice1_ = self.generate_np_indices(cut_name, prefix=1)
        slice2_ = self.generate_np_indices(cut_name, prefix=2)

        if self.save_fields:
            getattr(self, "j_3D"+name_append)[:] += X.j_3D[slice_] / self.nrep
            getattr(self, "Omega_pstxyz"+name_append)[:] += X.Omega_pstxyz[slice2_] / self.nrep
            getattr(self, "J_Omega_stxyz"+name_append)[:] += np.real(X.Omega_pstxyz[0][slice1_]* X.Omega_pstxyz[1][slice1_]) / self.nrep

        if self.save_populations or self.save_density_matrix:
            getattr(self, "rho_0_3D"+name_append)[:] += X.rho_0_3D[slice_] / self.nrep
        
        if self.save_density_matrix:
            getattr(self, "rho_ijtxyz"+name_append)[:] += X.rho_ijtxyz[slice_] / self.nrep

        elif self.save_populations:
            getattr(self, "rho_itxyz"+name_append)[:] += np.moveaxis(np.diagonal(X.rho_ijtxyz, axis1=0, axis2=1), -1, 0)[slice1_] / self.nrep

        if self.save_gain_polarization:
            getattr(self, "P_pstxyz"+name_append)[:] += X.P_pstxyz[slice2_] / self.nrep

        if self.save_combined_populations:
            getattr(self, "E_sstxyz"+name_append)[:] += X.E_sstxyz[slice2_] / self.nrep
            getattr(self, "G_sstxyz"+name_append)[:] += X.G_sstxyz[slice2_] / self.nrep

    def update_arrays_for_single_realization(self, X, rep_idx):
        """
        Updates all arrays with data for one realization of XLO_sim.

        Parameters
        ----------
        X : XLO_sim
            XLO_sim object for one realization of the simulation.

        rep_idx : int
            Index of the XLO_sim realization.

        Returns
        -------

        """
        if self.save_averages:
            self.nphoton_pump_z_avg[:] += X.nphoton_pump_z / self.nrep
            self.nphoton_sz_avg[:] += X.nphoton_sz / self.nrep
            self.nphoton_reg_sz_avg[:] += X.nphoton_reg_sz / self.nrep
        if self.save_individual_runs:
            self.nphoton_pump_z_reps[rep_idx] = X.nphoton_pump_z
            self.nphoton_sz_reps[rep_idx] = X.nphoton_sz
            self.nphoton_reg_sz_reps[rep_idx] = X.nphoton_reg_sz

        if self.save_full_grid:
            if self.save_averages:
                self.update_average_for_full_or_cut(X, "full")
            if self.save_individual_runs:
                self.update_arrays_for_full_or_cut(X, "full", rep_idx)

        elif self.save_data_cubes:
            cube_names = ["tcube", "xcube", "ycube", "zcube"]
            for i in range(4):
                if (cube_names[i] in self.cubes.keys()) and self.cubes[cube_names[i]]:
                    if self.save_averages:
                        self.update_average_for_cube(X, cube_names[i], i)
                    if self.save_individual_runs:
                        self.update_arrays_for_cube(X, cube_names[i], i, rep_idx)

        elif self.save_indexed_data:
            for cut_name in self.indices.keys():
                if self.save_averages:
                    self.update_average_for_full_or_cut(X, cut_name)
                if self.save_individual_runs:
                    self.update_arrays_for_full_or_cut(X, cut_name, rep_idx)

    def setup_file_attributes(self, file):
        """
        Adds attributes to the HDF5 file.

        Parameters
        ----------
        file : h5py.File
            File for storing the data.

        Returns
        -------

        """
        file.attrs["nrep"] = self.nrep
        for name in self.sim_attributes:
            file.attrs[name] = getattr(self.XLO_sim_reference, name)

    def save_dataset(self, group, data_name, name_append, is_average=False):
        """
        Creates or updates a dataset in the HDF5 file.

        Parameters
        ----------
        group : h5py.Group
            Subdirectory of the file where data is stored.

        data_name : string
            Name of the variable.

        name_append : string
            Suffix for the array names specifying the type of the arrays.

        is_average : bool
            Specifies wehther the data is average or for different realization of the simulations.

        Returns
        -------

        """
        data_array = getattr(self, data_name+name_append)

        if is_average:
            if data_name not in group:
                group.create_dataset(data_name, data=data_array)
            else:
                group[data_name][:] = (group[data_name][:] * self.prev_reps + data_array[:] * self.nrep) / (self.prev_reps + self.nrep)

        else:
            data_shape = data_array.shape
            if data_name not in group:
                group.create_dataset(data_name, data=data_array, maxshape=(None,)+data_shape[1:])
            else:
                group[data_name].resize((group[data_name].shape[0]+data_shape[0]), axis=0)
                group[data_name][-data_shape[0]:] = data_array[:]
            return

    def save_arrays_for_type(self, file, data_name, array_type):
        """
        Saves either average arrays or arrays for different realizations for the full grid, a data cube or a cut of indexed data to a HDF5 file. 

        Parameters
        ----------
        file : h5py.File
            File for storing the data.

        data_name : str
            Suffix for the array names specifying the type of the arrays.

        array_type : str
            "avg" if the data is average, or "reps" if it is the data for different realizations.

        Returns
        -------

        """
        if array_type == "avg":
            is_average=True
        else:
            is_average=False

        name_append = "_"+data_name+"_"+array_type
        group_name = array_type+"/"+data_name

        if group_name not in file:
            group = file.create_group(group_name)
            if self.save_data_cubes:
                group.attrs["indices"] = self.cubes[data_name]
            if self.save_indexed_data:
                group.attrs["indices"] = ', '.join(str(i) for i in self.indices[data_name])
        else:
            group = file[group_name]

        if self.save_fields:
            for variable_name in ["j_3D", "Omega_pstxyz", "J_Omega_stxyz"]:
                self.save_dataset(group, variable_name, name_append, is_average)

        if self.save_populations or self.save_density_matrix:
            self.save_dataset(group, "rho_0_3D", name_append, is_average)
        
        if self.save_density_matrix:
            self.save_dataset(group, "rho_ijtxyz", name_append, is_average)

        elif self.save_populations:
            self.save_dataset(group, "rho_itxyz", name_append, is_average)

        if self.save_gain_polarization:
            self.save_dataset(group, "P_pstxyz", name_append, is_average)

        if self.save_combined_populations:
            for variable_name in ["E_sstxyz", "G_sstxyz"]:
                self.save_dataset(group, variable_name, name_append, is_average)

    def save_arrays(self, file, name):
        """
        Saves both the average arrays and arrays for different realizations for the full grid, a data cube or a cut of indexed data to a HDF5 file. 

        Parameters
        ----------
        file : h5py.File
            File for storing the data.

        name : str
            Suffix for the array names specifying the type of the arrays.

        Returns
        -------

        """
        if self.save_averages:
            self.save_arrays_for_type(file, name, "avg")
        if self.save_individual_runs:
            self.save_arrays_for_type(file, name, "reps")

    def save_photons(self, file):
        """
        Saves the number of photons for both fields to a HDF5 file. 

        Parameters
        ----------
        file : h5py.File
            File for storing the data.

        Returns
        -------

        """
        if self.save_averages:
            if "avg" not in file:
                file.create_group("avg")
            self.save_dataset(file["avg"], "nphoton_pump_z", "_avg", is_average=True)
            self.save_dataset(file["avg"], "nphoton_sz", "_avg", is_average=True)
            self.save_dataset(file["avg"], "nphoton_reg_sz", "_avg", is_average=True)
        if self.save_individual_runs:
            if "reps" not in file:
                file.create_group("reps")
            self.save_dataset(file["reps"], "nphoton_pump_z", "_reps")
            self.save_dataset(file["reps"], "nphoton_sz", "_reps")
            self.save_dataset(file["reps"], "nphoton_reg_sz", "_reps")

    def check_parameters(self, attributes):
        """
        Checks whether the attributes of the HDF5 file to which the data is to be appended match the parameters of the current simulation run.

        Parameters
        ----------
        attibutes : dictionary
            Attributes of the existing HDF5 file.

        Returns
        -------
        bool
            True if the attributes in the file match with the parameters specified in reference XLO_sim object, false otherwise.

        """
        for name in self.sim_attributes:
            if getattr(self.XLO_sim_reference, name) != attributes[name]:
                return False
        return True

    def save_data(self):
        """
        Create a new HDF5 file or open the existing HDF5 file to which data is to be appended. In the latter case, check if the parameters of the current simulation match the attributes of the file. If they do not, create a new file. Save all data to file.

        Parameters
        ----------

        Returns
        -------

        """
        print("\nSaving data to file.")
        if self.h5_flag == "a":
            try:
                with h5py.File(self.data_file, "r") as file:
                    attributes = dict(file.attrs)
            except OSError:
                attributes = False

            if attributes:
                is_same_params = self.check_parameters(attributes)
                if is_same_params:
                    self.prev_reps = attributes["nrep"]
                    file_name = self.data_file
                    print(f"Appending data to {file_name}")
                else:
                    self.prev_reps = 0
                    time = datetime.now()
                    file_name = self.data_file[:-3]+"_"+time.strftime("%Y%m%d%H%M%S")+f"{randint(10)}"+".h5"
                    print(f"Simulation parameters don't match attributes from {self.data_file}... creating new file {file_name}")

            else:
                self.prev_reps = 0
                file_name = self.data_file
                print(f"Creating new file {file_name}")

        else:
            self.prev_reps = 0
            file_name = self.data_file
            print(f"Creating new file {file_name}")

        with h5py.File(file_name, self.h5_flag) as file:
            self.setup_file_attributes(file)
            self.save_photons(file)

            if self.save_full_grid:
                self.save_arrays(file, "full")

            elif self.save_data_cubes:
                for cube in self.cubes.keys():
                    if self.cubes[cube]:
                        self.save_arrays(file, cube)

            elif self.save_indexed_data:
                for cut in self.indices.keys():
                    self.save_arrays(file, cut)

            file.attrs["nrep"] = self.prev_reps + self.nrep
