module DoubleGyreInference

using HDF5, Statistics

export return_samples_file
export return_data_file

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

function return_samples_file(level_index, factor; sampled_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/", file_end_1 = "generative_samples.hdf5")
    filename = sampled_data_directory * return_prefix(level_index, factor) * file_end_1

    @info "loading samples from $filename"
    hfile = h5open(filename, "r")
    averaged_samples_1 = read(hfile["averaged samples 1"])
    averaged_samples_2 = read(hfile["averaged samples 2"])
    std_samples_1 = read(hfile["std samples 1"])
    std_samples_2 = read(hfile["std samples 2"])
    context_field_1 = read(hfile["context field 1"])
    context_field_2 = read(hfile["context field 2"])
    samples_1 = read(hfile["samples context 1"])
    samples_2 = read(hfile["samples context 2"])
    sample_index_1 = read(hfile["sample index 1"])
    sample_index_2 = read(hfile["sample index 2"])
    N = read(hfile["last training index"])
    close(hfile)
    return (; averaged_samples_1, averaged_samples_2, std_samples_1, std_samples_2, context_field_1, context_field_2, samples_1, samples_2, sample_index_1, sample_index_2)
end

function return_data_file(level_index; sample_index_1 = 4050, sample_index_2 = 3847, N = 3645, oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/")
    hfile = h5open(oceananigans_data_directory  * return_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
    total_field = read(hfile["field"])
    field_1 = total_field[:, :, :, sample_index_1]
    field_2 = total_field[:, :, :, sample_index_2]
    mean_field = mean(total_field[:, :,:,  1:N], dims = 4)[:, :, :, 1]
    std_field = std(total_field[:, :,:,  1:N], dims = 4)[:, :, :, 1]
    zlevel = read(hfile["zlevel"])

    b_mean =  read(hfile["b_mean"])
    b_2std = read(hfile["b_2std"])

    u_mean =  read(hfile["u_mean"])
    u_2std = read(hfile["u_2std"])

    v_mean =  read(hfile["v_mean"])
    v_2std = read(hfile["v_2std"])

    w_mean =  read(hfile["w_mean"])
    w_2std = read(hfile["w_2std"])
    close(hfile)
    return (; field_1, field_2, mean_field, std_field, zlevel, b_mean, b_2std, u_mean, u_2std, v_mean, v_2std, w_mean, w_2std)
end

end # module DoubleGyreInference
