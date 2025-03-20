using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
level_index = 1
hfile = h5open(oceananigans_data_directory  * DoubleGyreInference.return_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
total_field = read(hfile["field"])
close(hfile)

files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

sample_indices = collect(1:4050)
field_data = zeros(Float32, 128, 128, 4, length(files), length(sample_indices))
μs = zeros(4, length(files))
σs = zeros(4, length(files))
surface_data = zeros(Float32, 128, 128, length(sample_indices))
zlevels = zeros(length(files))

training_data = zeros(Float32, 128, 128, 61, length(sample_indices))
training_mus = zeros(Float32, 61)
training_stds = zeros(Float32, 61)


factor = 1
hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    data_tuple = return_data_file(level; complement = false, sample_index_2 = sample_indices )

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] # .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1))
    μs[:, i] .= μ
    σs[:, i] .= σ
    field_data[:, :, :, i, :] .= Float32.(field[:, :, 1:4, :])
    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (i, level) in ProgressBar(enumerate(levels_complement))
    ii = i + length(levels)
    data_tuple = return_data_file(level; complement = true, sample_index_2 = sample_indices )

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field_data[:, :, :, ii, :] .= Float32.(field[:, :, 1:4, :])
    μs[:, ii] .= μ
    σs[:, ii] .= σ
    zlevels[ii] = data_tuple.zlevel
end

dx = (dx[1:2:end] + dx[2:2:end])/2
dy = (dy[1:2:end] + dy[2:2:end])/2
dz = dz

permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_field_data = field_data[:, :, :, permuted_indices, :]
sorted_μs = μs[:, permuted_indices]
sorted_σs = σs[:, permuted_indices]

data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(data_directory_training * "eta_to_uvwb_at_z1_128_5.hdf5", "r")
eta_field = read(hfile["field"])[:, :, 5, sample_indices]
eta_mean = read(hfile["eta_mean"])
eta_2std = read(hfile["eta_2std"])
close(hfile)


training_data[:, :, 1:60, :] .= reshape(sorted_field_data[:, :, :, :, :], (128, 128, 60, length(sample_indices)))
training_data[:, :, 61, :] .= eta_field[:, :, :]
training_mus[1:60] .= Float32.(reshape(sorted_μs, 60))
training_stds[1:60] .= Float32.(reshape(sorted_σs, 60))
training_mus[61] = Float32(eta_mean)
training_stds[61] = Float32(eta_2std)


data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(data_directory_training  * "full_level_training_data.hdf5", "w")
write(hfile, "field", training_data)
write(hfile, "mus", training_mus)
write(hfile, "stds", training_stds)
close(hfile)