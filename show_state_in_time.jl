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

level_index = 3
M = 128 
case_var = 5
factor = 1
data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(data_directory_training * "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar).hdf5", "r")
field = read(hfile["field"])
eta_mean = read(hfile["eta_mean"])
eta_2std = read(hfile["eta_2std"])
close(hfile)

data_tuple = return_data_file(level_index)
scales = return_scale(data_tuple)
means = zeros(5)
stds = zeros(5)
means[1:4] .= scales[1][1:4]
stds[1:4] .= scales[2][1:4]
means[5] = eta_mean
stds[5] = eta_2std


lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)

fig = Figure(resolution = (1000, 800))
state_names = ["U", "V", "W", "T", "η"]
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color, free_surface_color]
for state_index in 1:5
    for (i, month) in enumerate([1000, 1001, 2000, 3000, 3645, 4000])
        ax = Axis(fig[state_index, i]; title = state_names[state_index] * " at Month " * string(month), xlabel = "Longitude", ylabel = "Latitude")
        if (state_index < 5) & (i > 1)
            hidedecorations!(ax)
        elseif (state_index < 5)
            hidexdecorations!(ax)
        elseif (i > 1)
            hideydecorations!(ax)
        else
            nothing
        end
        
        heatmap!(ax, lons, lats, field[:, :, state_index, month], colormap = colormaps[state_index], colorrange = (-1, 1))
    end
end
save("Figures/show_state_in_time_levelindex_$(level_index).png", fig)



fig = Figure(resolution = (1000, 800))
state_names = ["U", "V", "W", "T", "η"]
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color, free_surface_color]
for state_index in [5, 1, 2, 3, 4]
    for (i, month) in enumerate([240, 2400, 3600])
        year = month ÷ 12
        ax = Axis(fig[state_index, i]; title = state_names[state_index] * " at Year " * string(year), xlabel = "Longitude", ylabel = "Latitude")
        if (state_index < 5) & (i > 1)
            hidedecorations!(ax)
        elseif (state_index < 5)
            hidexdecorations!(ax)
        elseif (i > 1)
            hideydecorations!(ax)
        else
            nothing
        end
        
        heatmap!(ax, lons, lats, field[:, :, state_index, month], colormap = colormaps[state_index], colorrange = (-1, 1))
    end
end
save("Figures/show_state_in_time_levelindex_$(level_index)_year.png", fig)


factor = 75
fig = Figure(resolution = (12 * factor, 8 * factor))
state_names = ["U", "V", "W", "T", "η"]
state_units = ["[cm/s]", "[cm/s]", "[cm/s]", "[ᵒC]", "[m]"]
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color, free_surface_color]
years = [240, 2400, 3600]
ls = 20
for (j, state_index) in enumerate([5, 1, 2, 3, 4])
    for (i, month) in enumerate(years)
        year = month ÷ 12
        # title = state_names[state_index] * " at Year " * string(year),
        # , flip_ylabel = true
        ax = Axis(fig[i+1, j]; xlabel = "Longitude [ᵒ]", ylabel = "Year " * string(year) * " \n " * "Latitude [ᵒ]", ylabelsize = ls, xlabelsize = ls)
        if (i < 3) & (j > 1)
            hidedecorations!(ax)
        elseif (i < 3)
            hidexdecorations!(ax)
        elseif (j > 1)
            hideydecorations!(ax)
        else
            nothing
        end
        σ = stds[state_index]
        μ = means[state_index]
        if state_index == 4
            σ =  σ / (α * g)
            μ =  μ / (α * g)
            colorrange = (μ - σ, μ + σ)
        else
            colorrange = (-σ, σ)
        end
        heatmap!(ax, lons, lats, field[:, :, state_index, month] .* σ  .+ μ, colormap = colormaps[state_index], colorrange = colorrange)
    end
end
for (j, state_index) in enumerate([5, 1, 2, 3, 4])
    # ax = Axis(fig[4, j])
    σ = stds[state_index]
    μ = means[state_index]
    if state_index == 4
        σ =  σ / (α * g)
        μ =  μ / (α * g)
        colorrange = (μ - σ, μ + σ)
    elseif state_index == 5
        colorrange = (-σ, σ)
    else
        σ = σ  * 1e2 # cm/s
        colorrange = (-σ, σ)
    end  
    Colorbar(fig[1, j], label = state_names[state_index] * " " * state_units[state_index], colormap = colormaps[state_index], colorrange = colorrange, vertical = false)
end
save("Figures/show_state_in_time_levelindex_$(level_index)_year_T.png", fig)