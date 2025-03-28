module DoubleGyreInference

using HDF5, Statistics

export return_samples_file
export return_data_file
export return_scale

# jax version 
export jax_grab_scaling, jax_field, jax_context

const free_surface_color = :diverging_protanopic_deuteranopic_bwy_60_95_c32_n256
const temperature_color = :thermometer
const velocity_color = :balance

export free_surface_color, temperature_color, velocity_color

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

"""
    return_complement_prefix(level_index, factor; M = 128, casevar = 5)

Return the prefix of the file name for the data at a given vertical level and  coarse-graining factor.

# Arguments

- `level_index::Int`: the index of the vertical level.
- `factor::Int`: the coarse-graining factor.

# Optional arguments

- `M::Int = 128`: the number of grid points in the horizontal direction.
- `casevar::Int = 5`: the case variable.
"""
function return_complement_prefix(level_index, factor; M = 128, casevar = 5)
    return "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar)_complement_$(factor)_"
end

function return_samples_file(level_index, factor; sampled_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/", file_end_1 = "generative_samples.hdf5", complement = false)
    if complement
        filename = sampled_data_directory * return_complement_prefix(level_index, factor) * file_end_1
    else
        filename = sampled_data_directory * return_prefix(level_index, factor) * file_end_1
    end

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

function return_data_file(level_index; complement = false, sample_index_1 = 4050, sample_index_2 = 3847, N = 3645, oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/")
    if complement
        hfile = h5open(oceananigans_data_directory  * return_complement_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
    else
        hfile = h5open(oceananigans_data_directory  * return_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
    end
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

function return_scale(data_tuple)
    mean_scale = zeros(4)
    std_scale = zeros(4)
    mean_scale[1] = data_tuple.u_mean
    mean_scale[2] = data_tuple.v_mean
    mean_scale[3] = data_tuple.w_mean
    mean_scale[4] = data_tuple.b_mean
    std_scale[1] = data_tuple.u_2std
    std_scale[2] = data_tuple.v_2std
    std_scale[3] = data_tuple.w_2std
    std_scale[4] = data_tuple.b_2std
    return mean_scale, std_scale
end


function jax_grab_scaling(state_index)
    data_file = "full_level_training_data.hdf5"
    data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
    hfile = h5open(data_directory * data_file, "r")
    mu = read(hfile["mus"])[state_index+1]
    std = read(hfile["stds"])[state_index+1]
    close(hfile)
    return (; mu, std)
end

function jax_field(level::Int, field_index::Int, future_year::Int; data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/", file_string = "production_jax_samples_", cg = -1)
    state_index = field_index + (level-1) * 4
    if cg < 0
        total_string = data_directory * file_string * "$(future_year)_field_$(state_index).hdf5"
    else
        total_string = data_directory * file_string * "$(future_year)_field_$(state_index)_cg_$(cg).hdf5"
    end
    hfile = h5open(total_string, "r")
    ground_truth = read(hfile["ground_truth"])
    samples = read(hfile["samples"])
    close(hfile)
    scaling = jax_grab_scaling(state_index)
    mu = scaling.mu 
    sigma = scaling.std
    return (; ground_truth, samples, mu, sigma)
end

function jax_context(future_year::Int; data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/", file_string = "production_jax_samples_", cg = -1)
    if cg < 0
        total_string = data_directory * file_string * "$(future_year)_field_0.hdf5"
    else
        total_string = data_directory * file_string * "$(future_year)_field_0_cg_$(cg).hdf5"
    end
    hfile = h5open(total_string , "r")
    context = read(hfile["context"])
    close(hfile)
    (; mu, std) = jax_grab_scaling(60)
    sigma = std
    return (; context, mu, sigma)
end

function jax_symbol_to_index(field_symbol::Symbol)
    if field_symbol == :u 
        return 0
    elseif field_symbol == :v 
        return 1 
    elseif field_symbol == :w 
        return 2 
    else 
        return 3 
    end
end

function jax_field(level, field_symbol::Symbol, future_year; data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/", file_string = "production_jax_samples_", cg = false)
    jax_field(level, jax_symbol_to_index(field_symbol), future_year; data_directory, file_string, cg) 
end

end # module DoubleGyreInference
