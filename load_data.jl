using HDF5

sampled_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"
oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
figure_directory = "Figures/"

"""
    return_prefix(level_index, factor; M = 128, casevar = 5)

Return the prefix of the file name for the data at a given vertical level and  coarse-graining factor.

# Arguments

- `level_index::Int`: the index of the vertical level.
- `factor::Int`: the coarse-graining factor.

# Optional arguments

- `M::Int = 128`: the number of grid points in the horizontal direction.
- `casevar::Int = 5`: the case variable.
"""
function return_prefix(level_index, factor; M = 128, casevar = 5)
    return "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar)_$(factor)_"
end