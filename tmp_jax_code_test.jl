using HDF5

function jax_grab_scaling(state_index)
    data_file = "full_level_training_data.hdf5"
    data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
    hfile = h5open(data_directory * data_file, "r")
    mu = read(hfile["mus"])[state_index+1]
    std = read(hfile["stds"])[state_index+1]
    close(hfile)
    return (; mu, std)
end

function jax_field(level::Int, field_index::Int, future_year::Int; data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/")
    state_index = field_index + (level-1) * 4
    hfile = h5open(data_directory * "production_jax_samples_$(future_year)_field_$(state_index).hdf5", "r")
    ground_truth = read(hfile["ground_truth"])
    samples = read(hfile["samples"])
    close(hfile)
    scaling = jax_grab_scaling(state_index)
    mu = scaling.mu 
    std = scaling.std
    return (; ground_truth, samples, mu, std)
end

function jax_context(future_year::Int; data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/")
    hfile = h5open(data_directory*"production_jax_samples_$(future_year)_field_0.hdf5", "r")
    context = read(hfile["context"])
    close(hfile)
    (; mu, std) = jax_grab_scaling(60)
    return (; context, mu, std)
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

jax_field(level, field_symbol::Symbol, future_year)=jax_field(level, jax_symbol_to_index(field_symbol), future_year) 