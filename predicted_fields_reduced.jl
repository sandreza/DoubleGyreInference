using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5
M = 256 
casevar = 5
α = 2e-4
g = 9.81
Nz = 15
Lz = 1800
σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
N1 = 1126 # start of the training data
N2 = 3645 # end of the training data

levels = [1, 2, 3, 5, 7] 
zlevels = z_centers.(1:Nz)
level = 3
depth_string = @sprintf("%0.f", abs(zlevels[[3, collect(9:14)...][level]]))
data_tuple = return_data_file(level)
sample_tuples = []
for cg in [1, 2, 4, 8, 16, 32, 64, 128]
    push!(sample_tuples, return_samples_file(level, cg)) 
end

field = data_tuple.field_2   

fig = Figure(resolution = (1300, 600) )
state_index = 1
state_names = ["U", "V", "W", "T"]
units = ["m/s", "m/s", "m/s", "K"]
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color]
cg_ind = 1
cranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
for (ii, i) in enumerate([1, 2, 4]) 
    ax = Axis(fig[ii, 1]; title = "Free Surface Height") 
    hidedecorations!(ax)
    cf = sample_tuples[cg_ind].context_field_2
    heatmap!(ax, lons, lats, cf[:, :, 1], colormap = free_surface_color, colorrange = (-1, 1))
    ax = Axis(fig[ii, 2]; title = "Oceananigans " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, field[:, :, i], colormap = colormaps[i], colorrange = cranges[i])
    ax = Axis(fig[ii, 3]; title = "AI Average " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].averaged_samples_2[:, :, i], colormap = colormaps[i], colorrange = cranges[i])
    ax = Axis(fig[ii, 4]; title = "Difference " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, field[:, :, i] - sample_tuples[cg_ind].averaged_samples_2[:, :, i], colormap = :balance, colorrange = (-1, 1))
    ax = Axis(fig[ii, 5]; title = "AI Uncertainty " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, i], colormap = :viridis, colorrange = (0, 1/4))
    ax = Axis(fig[ii, 6]; title = "AI Sample 1 " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, i, 1], colormap = colormaps[i], colorrange = cranges[i])
    ax = Axis(fig[ii, 7]; title = "AI Sample 2 " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, i, 2], colormap = colormaps[i], colorrange = cranges[i])
end
save("Figures/context_field_and_prediction_states_reduced.png", fig)