using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

vlevels_data = zeros(128, 128, length(files))
wlevel_data = zeros(128, 128, length(files))
wlevels_samples = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))
mbfield = zeros(128, 128, 15)
local_field = zeros(128, 128, 15, 4)


levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    data_tuple = return_data_file(level; complement = false)

    μ, σ = return_scale(data_tuple)
    mfield = data_tuple.mean_field
    mfield[:, :, 1:4] .= mfield[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    mbfield[:, :, i] .= mfield[:, :, 4]
    local_field[:, :, i, 1:4] .= data_tuple.field_2[:,:, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))

    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (i, level) in ProgressBar(enumerate(levels_complement))
    ii = i + length(levels)
    data_tuple = return_data_file(level; complement = true)

    #=
    oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
    hfile = h5open(oceananigans_data_directory  * DoubleGyreInference.return_complement_prefix(level, 1)[1:end-3] * ".hdf5", "r")
    N = 3645
    mean_field = mean(read(hfile["field"])[:, :, :, 1:N], dims = 4)[:, :, :, 1]
=#
    μ, σ = return_scale(data_tuple)
    mfield = data_tuple.mean_field
    mfield[:, :, 1:4] .= mfield[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    mbfield[:, :, ii] .= mfield[:, :, 4]
    local_field[:, :, ii, 1:4] .= data_tuple.field_2[:,:, 1:4]  .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    zlevels[ii] = data_tuple.zlevel
end



permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
mbfield_sorted = mbfield[:, :, permuted_indices]
local_field_sorted = local_field[:, :, permuted_indices, :]

level_index = 8
oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(oceananigans_data_directory  * DoubleGyreInference.return_complement_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
sample_index_2 = 3847
total_field = read(hfile["field"])
sigma =  read(hfile["eta_2std"])
mu = read(hfile["eta_mean"])
eta = total_field[:, :, 5, sample_index_2] .* sigma  .+ mu
close(hfile)

hfile = h5open(oceananigans_data_directory * "theory_data.hdf5", "w")
hfile["b"] = mbfield_sorted
hfile["z"] = sorted_zlevels
hfile["u prediction"] = local_field_sorted[:, :, :, 1]
hfile["v prediction"] = local_field_sorted[:, :, :, 2]
hfile["w prediction"] = local_field_sorted[:, :, :, 3]
hfile["b prediction"] = local_field_sorted[:, :, :, 4]
hfile["eta for prediction"] = eta
close(hfile)