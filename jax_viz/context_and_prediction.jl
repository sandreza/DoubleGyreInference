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

    shift2 = 1
    ax = Axis(fig[ii, 6+shift1 + shift2]; title = "AI Uncertainty " * state_names[i])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, i] .* σ[i], colormap = :viridis, colorrange = (0, 1/4) .* σ[i])
    Colorbar(fig[ii, 7 + shift1 + shift2], hm, label = units[i])
end
save("Figures/context_field_and_prediction_states_reduced.png", fig)

field = data_tuple.field_2   
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color]
colormaps = [velocity_color, velocity_color, velocity_color, temperature_color]

units = [L"\text{cm/s}", L"\text{cm/s}", L"\text{mm/s}", L"^\circ\text{C}"]

meanticks = (([-0.03, 0, 0.03], [L"-3", L"0", L"3"]),
            ([-0.05, 0, 0.05], [L"-5", L"0", L"5"]),
            ([-5e-5, 0, 5e-5], [L"-0.005", L"0", L"0.005"]),
            ([0, 5, 10], [L"0", L"5", L"10"]))

stdticks = ([0, 0.005, 0.01], [L"0", L"0.5", L"1"]),
           ([0, 0.005, 0.01, 0.015, 0.02], [L"0", L"0.5", L"1", L"1.5", L"2"]),
           ([0, 5e-6, 1e-5, 1.5e-5], [L"0", L"0.005", L"0.01", L"0.015"]),
           ([0, 0.5, 1], [L"0", L"0.5", L"1"])

function context_field_prediction!(fig, i, cg_ind, t)
    if i == 1
        ax = Axis(fig[i, 1]; title = L"\text{\textbf{Free Surface Height}}")
    else
        ax = Axis(fig[i, 1];)
    end
    hidedecorations!(ax) 
    cf = sample_tuples[cg_ind].context_field_2
    hm = heatmap!(ax, lons, lats, cf[:, :, 1] .* eta_sigma .+ eta_mu, colormap = free_surface_color, colorrange = (-1, 1))
    Colorbar(fig[i, 2], hm, label = L"\text{m}", ticks=([-1, -0.5, 0, 0.5, 1], [L"-1", L"-0.5", L"0", L"0.5", L"1"]), labelsize=28, ticklabelsize=28)

    shift1 = 1
    if i == 1
        ax = Axis(fig[i, 2 + shift1]; title = L"\text{\textbf{Oceananigans}}")
    else
        ax = Axis(fig[i, 2 + shift1];)
    end
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, field[:, :, t] .* σ[t] .+ μ[t], colormap=colormaps[t], colorrange = cranges[t])
    if i == 1
        ax = Axis(fig[i, 3 + shift1]; title = L"\text{\textbf{AI Average}}")
    else
        ax = Axis(fig[i, 3 + shift1];)
    end
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].averaged_samples_2[:, :, t] .* σ[t] .+ μ[t], colormap=colormaps[t], colorrange = cranges[t])
    if i == 1
        ax = Axis(fig[i, 4 + shift1]; title = L"\text{\textbf{AI Sample 1}}")
    else
        ax = Axis(fig[i, 4 + shift1];)
    end
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, t, 10 * i] .* σ[t] .+ μ[t], colormap=colormaps[t], colorrange = cranges[t])
    if i == 1
        ax = Axis(fig[i, 5 + shift1]; title = L"\text{\textbf{Data mean}}")
    else
        ax = Axis(fig[i, 5 + shift1];)
    end
    hidedecorations!(ax)
    # Removed extra sample plot
    # heatmap!(ax, lons, lats, sample_tuples[cg_ind].samples_2[:, :, t, 20 * i] .* σ[t] .+ μ[t], colormap=colormaps[t], colorrange = cranges[t])
    # if i == 1
    #     ax = Axis(fig[i, 6 + shift1]; title = L"\text{\textbf{Data Mean}}")
    # else
    #     ax = Axis(fig[i, 6 + shift1];)
    # end
    # hidedecorations!(ax)
    heatmap!(ax, lons, lats, data_tuple.mean_field[:, :, t] .* σ[t] .+ μ[t], colormap = colormaps[t], colorrange = cranges[t])
    Colorbar(fig[i, 6 + shift1], hm, label=units[t], ticks=meanticks[t], labelsize=28, ticklabelsize=28)

    if i == 1
        ax = Axis(fig[i, 8]; title = L"\text{\textbf{AI Uncertainty}}")
    else
        ax = Axis(fig[i, 8];)
    end

    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, t] .* σ[t] , colormap = :viridis, colorrange = ((0, 1/4) .* σ[t]))
    if i == 1
        ax = Axis(fig[i, 9]; title =  L"\text{\textbf{Data Std}}")
    else
        ax = Axis(fig[i, 9];)
    end
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, data_tuple.std_field[:, :, t] .* σ[t], colormap = :viridis, colorrange = ((0, 1/4) .* σ[t]))
    Colorbar(fig[i, 10], hm, label = units[t], ticks=stdticks[t], labelsize=28, ticklabelsize=28)

    return fig
end

####
#### U - velocity figure
####

fig = Figure(size = (2400, 1200), fontsize=28)
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
for (i, cg_ind) in enumerate([1, 3, 5, 7])
    context_field_prediction!(fig, i, cg_ind, 1)
end
save("Figures/context_field_and_prediction_velocity.png", fig)
save("Figures/context_field_and_prediction_velocity.eps", fig)

####
#### Temperature figure
####

fig = Figure(size = (2400, 1200), fontsize=28)
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
for (i, cg_ind) in enumerate([1, 3, 5, 7])
    context_field_prediction!(fig, i, cg_ind, 4)
end
save("Figures/context_field_and_prediction_temperature.png", fig)
save("Figures/context_field_and_prediction_temperature.eps", fig)

####
#### All fields figure
####

fig = Figure(size = (2400, 1200), fontsize=28)
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
cg_ind = 1
for i in [1, 2, 3, 4]
    context_field_prediction!(fig, i, cg_ind, i)
end
save("Figures/context_field_and_prediction_all.png", fig)
save("Figures/context_field_and_prediction_all.eps", fig)

t = 4
fig = Figure(size = (2400, 400), fontsize=28)
lats = range(15, 75, length = 128)
lons = range(0, 60, length = 128)
cg_ind = 7

ax = Axis(fig[1, 1]; title = L"\text{\textbf{AI Average}}")
heatmap!(ax, lons, lats, sample_tuples[cg_ind].averaged_samples_2[:, :, t] .* σ[t] .+ μ[t], colormap=colormaps[t], colorrange = cranges[t])
hidedecorations!(ax)
ax = Axis(fig[1, 2]; title = L"\text{\textbf{Data Mean}}")
hidedecorations!(ax)
hm = heatmap!(ax, lons, lats, data_tuple.mean_field[:, :, t] .* σ[t] .+ μ[t], colormap = colormaps[t], colorrange = cranges[t])
Colorbar(fig[1, 3], hm, label=units[t], ticks=meanticks[t], labelsize=28, ticklabelsize=28)

shift =1 
Δ =  abs.(sample_tuples[cg_ind].averaged_samples_2[:, :, t] - data_tuple.mean_field[:, :, t])
ax = Axis(fig[1, 4] ; title = L"\text{\textbf{Absolute Difference}}")
heatmap!(ax, lons, lats, Δ .* σ[t], colormap = :viridis, colorrange = (0, 1/4) .* σ[t])
hidedecorations!(ax)
ax = Axis(fig[1, 5]; title =  L"\text{\textbf{AI Uncertainty}}")
hidedecorations!(ax)
hm = heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, t] .* σ[t] , colormap = :viridis, colorrange = ((0, 1/4) .* σ[t]))
ax = Axis(fig[1, 6]; title =  L"\text{\textbf{Sum}}")
hidedecorations!(ax)
hm = heatmap!(ax, lons, lats, sample_tuples[cg_ind].std_samples_2[:, :, t] .* σ[t] + Δ .* σ[t], colormap = :viridis, colorrange = ((0, 1/4) .* σ[t]))
ax = Axis(fig[1, 7]; title =  L"\text{\textbf{Data Standard Deviation}}")
hidedecorations!(ax)
hm = heatmap!(ax, lons, lats, data_tuple.std_field[:, :, t] .* σ[t], colormap = :viridis, colorrange = ((0, 1/4) .* σ[t]))
Colorbar(fig[1, 8], hm, label = units[t], ticks=stdticks[t], labelsize=28, ticklabelsize = 28)

save("Figures/context_field_and_prediction_temperature_difference.png", fig)