using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5, Random
using LaTeXStrings
α = 2e-4
g = 9.81

level_indices = [14, 12, 10, 3]
total_levels = length(level_indices)
Nz = 15
Lz = 1800
nsamples = 100

cg_levels = 8
v = zeros(128, 128)
T  = zeros(128, 128)
u = zeros(128, 128)
w = zeros(128, 128)
T_samples = zeros(128, 128, nsamples, cg_levels)
v_samples = zeros(128, 128, nsamples, cg_levels)
w_samples = zeros(128, 128, nsamples, cg_levels)
u_samples = zeros(128, 128, nsamples, cg_levels)
eta = zeros(128, 128, cg_levels)


future_year = 50
file_string = "attention_velocity_uc_production_jax_samples_"

σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevels = z_centers.(1:Nz)
level = 10
scales_factor = [1e-2, 1e-2, 1e-6, (α * g) ]
for cg in 0:7
    (; ground_truth, samples, mu, sigma) = jax_field(level, :u, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu)  / scales_factor[1]
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)  / scales_factor[1]

    directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/" 
    new_file_string = "_traditional_cg_$(cg).hdf5"
    state_index = 0
    field_index = 4 * (level - 1) + state_index + 1
    hfile = h5open(directory * new_file_string, "r")
    mean_sample = read(hfile["output"])[:, :, field_index]
    ground_truth_2 = read(hfile["ground_truth"])[:, :, field_index]
    u_mean = (mean_sample .* sigma .+ mu ) / scales_factor[1]
    close(hfile)

    lats = range(15, 75, length = 128)  # latitude range for the heatmap
    lons = range(0, 60, length = 128)

    crange = quantile(u[:], 0.99)

    fig = Figure(resolution = (1250, 250))
    generative_mean = mean(u_samples[:, :, :, cg + 1], dims = 3)[:, :, 1]
    ax = Axis(fig[1, 1]; title = "Truth")
    heatmap!(ax, lons, lats, u, colorrange = (-crange, crange), colormap = velocity_color)
    ax = Axis(fig[1, 2]; title = "Generative Mean")
    heatmap!(ax, lons, lats, generative_mean , colorrange = (-crange, crange), colormap = velocity_color)
    ax = Axis(fig[1, 3]; title = "Traditional Model")
    hm = heatmap!(ax, lons, lats, u_mean, colorrange = (-crange, crange), colormap = velocity_color)
    Colorbar(fig[1, 4], hm, label = "u (m/s)")
    ax = Axis(fig[1, 5]; title = "Generative Mean - Truth")
    heatmap!(ax, lons, lats, abs.(generative_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    ax = Axis(fig[1, 6]; title = "Traditional Model - Truth")
    hm = heatmap!(ax, lons, lats, abs.(u_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    Colorbar(fig[1, 7], hm, label = "u (m/s)")
    println("The mean absolute error of the generative mean is ", mean(abs.(generative_mean - u)))
    println("The mean absolute error of the traditional mean is ", mean(abs.(u_mean - u)))
    save("Figures/deterministic_$cg.png", fig)
end
