using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

level_index = 8
oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(oceananigans_data_directory  * DoubleGyreInference.return_complement_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
eta_sigma =  read(hfile["eta_2std"])
eta_mu = read(hfile["eta_mean"])
close(hfile)

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

μ, σ = return_scale(data_tuple)

fig = Figure(resolution = (1300, 600) )
state_index = 1
state_names = ["U", "V", "W", "T"]
units = ["m/s", "m/s", "m/s", "°C"]
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color]
cg_ind = 1
cranges = [(-(μ[i] + σ[i]), μ[i] + σ[i]) for i in 1:4]

α = 2e-4
g = 9.81

μ[4] /= (α * g)
σ[4] /= (α * g)
cranges[4] = (0, 10)
for (ii, i) in enumerate([1, 2, 4]) 
    if ii == 2
        ax = Axis(fig[ii, 1]; title = "Free Surface Height") 
        hidedecorations!(ax)
        cf = sample_tuples[cg_ind].context_field_2
        hm = heatmap!(ax, lons, lats, cf[:, :, 1] .* eta_sigma .+ eta_mu, colormap = free_surface_color, colorrange = (-1, 1))
        Colorbar(fig[ii, 2], hm, label = "m")
    end
    shift1 = 1
    ax = Axis(fig[ii, 2+shift1]; title = "Oceananigans " * state_names[i])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, field[:, :, i] .* σ[i] .+ μ[i], colormap = colormaps[i], colorrange = cranges[i])
    ax = Axis(fig[ii, 3+shift1]; title = "AI Average " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].averaged_samples_2[:, :, i] .* σ[i] .+ μ[i], colormap = colormaps[i], colorrange = cranges[i])

    ax = Axis(fig[ii, 4+shift1]; title = "AI Sample 1 " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, i, 1].* σ[i] .+ μ[i], colormap = colormaps[i], colorrange = cranges[i])
    ax = Axis(fig[ii, 5+shift1]; title = "AI Sample 2 " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, i, 2].* σ[i] .+ μ[i], colormap = colormaps[i], colorrange = cranges[i])
    Colorbar(fig[ii, 6 + shift1], hm, label = units[i])

    #=
    ax = Axis(fig[ii, 4+shift1 + shift2]; title = "Difference " * state_names[i])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, field[:, :, i] - sample_tuples[cg_ind].averaged_samples_2[:, :, i], colormap = :balance, colorrange = ((-1, 1) .* σ[i] .+ μ[i]))
    =#
    shift2 = 1
    ax = Axis(fig[ii, 6+shift1 + shift2]; title = "AI Uncertainty " * state_names[i])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, i] .* σ[i], colormap = :viridis, colorrange = (0, 1/4) .* σ[i])
    Colorbar(fig[ii, 7 + shift1 + shift2], hm, label = units[i])

end
save("Figures/context_field_and_prediction_states_reduced.png", fig)


field = data_tuple.field_2   

fig = Figure(resolution = (1800, 450))
state_index = 1
state_names = ["U", "V", "W", "T"]
units = ["m/s", "m/s", "m/s", "°C"]
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color]
for (i, cg_ind) in enumerate([1, 5])
    ax = Axis(fig[i, 1]; title = "Free Surface Height")
    hidedecorations!(ax) 
    cf = sample_tuples[cg_ind].context_field_2
    hm = heatmap!(ax, lons, lats, cf[:, :, 1] .* eta_sigma .+ eta_mu, colormap = free_surface_color, colorrange = (-1, 1))
    Colorbar(fig[i, 2], hm, label = "m", labelsize = 20, ticklabelsize = 20)

    shift1 = 1
    ax = Axis(fig[i, 2 + shift1]; title = "Oceananigans ")
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, field[:, :, 1] .* σ[1] .+ μ[1], colormap = :balance, colorrange = cranges[1])
    ax = Axis(fig[i, 3 + shift1]; title = "AI Average ")
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].averaged_samples_2[:, :, 1] .* σ[1] .+ μ[1], colormap = :balance, colorrange = cranges[1])
    ax = Axis(fig[i, 4 + shift1] ; title = "AI Sample 1")
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, 1, 10 * i] .* σ[1] .+ μ[1], colormap = :balance, colorrange = cranges[1])
    ax = Axis(fig[i, 5 + shift1]; title = "AI Sample 2")
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, 1, 20 * i] .* σ[1] .+ μ[1], colormap = :balance, colorrange = cranges[1])
    ax = Axis(fig[i, 6 + shift1]; title = "Data Mean")
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, data_tuple.mean_field[:, :, 1] .* σ[1] .+ μ[1], colormap = :balance, colorrange = cranges[1])
    Colorbar(fig[i, 7 + shift1], hm, label = "m/s", labelsize = 20, ticklabelsize = 20)

    ax = Axis(fig[i, 9]; title = "AI Uncertainty ")
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, 1] .* σ[1] , colormap = :viridis, colorrange = ((0, 1/4) .* σ[1]))
    ax = Axis(fig[i, 10]; title = "Data Standard Deviation")
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, data_tuple.std_field[:, :, 1] .* σ[1], colormap = :viridis, colorrange = ((0, 1/4) .* σ[1]))
    Colorbar(fig[i, 11], hm, label = "m/s", labelsize = 20, ticklabelsize = 20)
end
save("Figures/context_field_and_prediction_reduced.png", fig)