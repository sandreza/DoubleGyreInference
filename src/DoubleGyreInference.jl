module DoubleGyreInference

greet() = print("Hello World!")

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

end # module DoubleGyreInference
