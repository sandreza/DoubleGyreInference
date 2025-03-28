using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5, Random
α = 2e-4
g = 9.81

level_indices = [14, 12, 10, 3]
total_levels = length(level_indices)
Nz = 15
Lz = 1800
nsamples = 100

cg_levels = 8
sorted_vlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_data  = zeros(128, 128, total_levels)
sorted_ulevels_data = zeros(128, 128, total_levels)
sorted_wlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)
sorted_vlevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)
sorted_wlevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)
sorted_ulevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)

future_year = 50
cg = 0 # coarse-graining level
file_string = "attention_velocity_uc_production_jax_samples_"

σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevels = z_centers.(1:Nz)

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dz = read(hfile["dz"])
close(hfile)

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/heat_flux_in_time.hdf5", "r")
hflux = read(hfile["heat_flux"])
close(hfile)

for cg in 0:(cg_levels-1)
    for level in ProgressBar(1:total_levels)
        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :u, future_year; file_string, cg)
        sorted_ulevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
        sorted_ulevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu) 

        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :v, future_year; file_string, cg)
        sorted_vlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
        sorted_vlevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu) 

        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :w, future_year; file_string, cg)
        sorted_wlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu)
        sorted_wlevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu)

        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :b, future_year; file_string, cg)
        sorted_Tlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) /(α * g)
        sorted_Tlevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu) /(α * g)
    end
end


ground_truth_error_u = zeros(total_levels, cg_levels, 100)
ground_truth_error_v = zeros(total_levels, cg_levels, 100)
ground_truth_error_w = zeros(total_levels, cg_levels, 100)
ground_truth_error_T = zeros(total_levels, cg_levels, 100)

shuffle_error_u = zeros(total_levels, cg_levels, 100)
shuffle_error_v = zeros(total_levels, cg_levels, 100)
shuffle_error_w = zeros(total_levels, cg_levels, 100)
shuffle_error_T = zeros(total_levels, cg_levels, 100)
scales = [1e-2, 1e-2, 1e-3, 1]
for cg in 0:(cg_levels-1)
    for level in 1:total_levels
        ground_truth_error_u[level, cg + 1, :] = mean(abs.( sorted_ulevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_ulevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[1]
        ground_truth_error_v[level, cg + 1, :] = mean(abs.( sorted_vlevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_vlevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[2]
        ground_truth_error_w[level, cg + 1, :] = mean(abs.( sorted_wlevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_wlevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[3]
        ground_truth_error_T[level, cg + 1, :] = mean(abs.( sorted_Tlevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_Tlevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[4]

        shuffle_error_u[level, cg + 1, :] = mean(abs.(sorted_ulevels_samples[:, :, level, perm, cg + 1] - sorted_ulevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[1]
        shuffle_error_v[level, cg + 1, :] = mean(abs.(sorted_vlevels_samples[:, :, level, perm, cg + 1] - sorted_vlevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[2]
        shuffle_error_w[level, cg + 1, :] = mean(abs.(sorted_wlevels_samples[:, :, level, perm, cg + 1] - sorted_wlevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[3]
        shuffle_error_T[level, cg + 1, :] = mean(abs.(sorted_Tlevels_samples[:, :, level, perm, cg + 1] - sorted_Tlevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[4]
    end
end


tmpu = mean(ground_truth_error_u, dims = 3)