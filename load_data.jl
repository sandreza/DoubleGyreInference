using HDF5, DoubleGyreInference, Statistics

r_pref = DoubleGyreInference.return_prefix
sampled_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"
oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
file_end_1 = "generative_samples.hdf5"
field_end_2 = "double_gyre_losses.hdf5"
figure_directory = "Figures/"
isdir(figure_directory) || mkdir(figure_directory)

level_index = 3
filename = sampled_data_directory * r_pref(level_index, 1) * file_end_1

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
@info "loading field"
hfile = h5open(oceananigans_data_directory  * r_pref(level_index, 1)[1:end-3] * ".hdf5", "r")
total_field = read(hfile["field"])
field_1 = [:, :, :, sample_index_1]
field_2 = read(hfile["field"])[:, :, :, sample_index_2]
mean_field = mean(read(hfile["field"])[:, :,:,  1:N], dims = 4)
std_field = std(read(hfile["field"])[:, :,:,  1:N], dims = 4)
zlevel = read(hfile["zlevel"])
close(hfile)
tupled_data = (; field_1, field_2, averaged_samples_1, averaged_samples_2, std_samples_1, std_samples_2, context_field_1, context_field_2, samples_1, samples_2, sample_index_1, sample_index_2)