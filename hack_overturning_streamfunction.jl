using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

α = 2e-4
g = 9.81

factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

vlevels_data = zeros(128, 128, length(files))
Tlevel_data = zeros(128, 128, length(files))
Tlevels_samples = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))


levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    sample_tuple = return_samples_file(level, factor; complement = false)
    data_tuple = return_data_file(level; complement = false)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    Tlevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, i] .= field[:, :, 2]
    Tlevel_data[:, :, i] .= field[:, :, 4]

    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (i, level) in ProgressBar(enumerate(levels_complement))
    ii = i + length(levels)

    sample_tuple = return_samples_file(level, factor; complement = true)
    data_tuple = return_data_file(level; complement = true)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    vlevels_samples[:, :, ii, :] .= average_samples[:, :, 2, :]
    Tlevels_samples[:, :, ii, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, ii] .= field[:, :, 2]
    Tlevel_data[:, :, ii] .= field[:, :, 4]
    
    zlevels[ii] = data_tuple.zlevel
end

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)


permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_vlevels_data = vlevels_data[:, :, permuted_indices]
sorted_Tlevel_data = Tlevel_data[:, :, permuted_indices]
sorted_vlevels_samples = vlevels_samples[:, :, permuted_indices, :]
sorted_Tlevels_samples = Tlevels_samples[:, :, permuted_indices, :]

∂z = [sorted_zlevels[i+1] - sorted_zlevels[i] for i in 1:14]

Tz = (sorted_Tlevel_data[:, :, 2:end] - sorted_Tlevel_data[:, :, 1:end-1] ) ./ reshape(∂z, (1, 1, 14))
Tz_samples = (sorted_Tlevels_samples[:, :, 2:end, :] - sorted_Tlevels_samples[:, :, 1:end-1, :] ) ./ reshape(∂z, (1, 1, 14, 1))
Tz_bar = mean(Tz, dims = 1)
Tz_samples_bar = mean(Tz_samples, dims = 1)
Tz_samples_bar_average = mean(Tz_samples_bar, dims = 4)[1, :, :, 1]
Tz_samples_bar_std = std(Tz_samples_bar, dims = 4)[1, :, :, 1]

rad_to_deg = π / 180
dλ = 60 / 128 * rad_to_deg
Lλ = 60 * rad_to_deg
# comment vbar uses the mean, technically we need to  sum, we multiply by the factor here 
ρ = 1000 
c_p = 4000
vfactor = reshape(cosd.(range(15, 75, length = 128)), (1, 128,1, 1)) * 40007863 * ρ * c_p / 10^(15) / (2π)


vbar_data = mean(sorted_vlevels_data, dims = 1)
Tbar_data = mean(sorted_Tlevel_data, dims = 1)
vbar_samples = mean(sorted_vlevels_samples, dims = 1)
Tbar_samples = mean(sorted_Tlevels_samples, dims = 1)
vpTp_data = mean((sorted_vlevels_data .- vbar_data) .* (sorted_Tlevel_data .- Tbar_data), dims = 1)
vpTp_samples = mean((sorted_vlevels_samples .- vbar_samples) .* (sorted_Tlevels_samples .- Tbar_samples), dims = 1) 
ensemble_mean_vT_samples = mean(vpTp_samples, dims = 4)[1, :, :, 1] 
ensemble_std_vT_samples = std(vpTp_samples, dims = 4)[1, :, :, 1] 
minstrat = 0e-3
hack_qgΨ_samples = mean((vpTp_samples[:, :, 2:end, :] + vpTp_samples[:, :, 2:end, :])/2, dims = 1) ./ (mean(Tz_samples, dims = 1) .+ minstrat)
hack_qgΨ_samples_mean = mean(hack_qgΨ_samples, dims = 4)[1, :, :, 1]
hack_qgΨ_samples_std = std(hack_qgΨ_samples, dims = 4)[1, :, :, 1]
hack_qgΨ_data = mean((vpTp_data[:, :, 2:end] + vpTp_data[:, :, 2:end])/2, dims = 1) ./ (Tz_bar .+ minstrat)
hack_qgΨ_data = hack_qgΨ_data[1, :, :]



latitude = range(15, 75, length = 128)
fig = Figure() 
ax = Axis(fig[1, 1])
colorrange = extrema(vpTp_data)
heatmap!(ax, latitude, sorted_zlevels, vpTp_data[1, :, :], colorrange = colorrange)
ax = Axis(fig[1, 2])
heatmap!(ax, latitude, sorted_zlevels, ensemble_mean_vT_samples, colorrange = colorrange)
save("Figures/vpTp.png", fig)

latitude = range(15, 75, length = 128)
fig = Figure(resolution = (1200, 400)) 
ax = Axis(fig[1, 1]; title = "QG Ψ: Oceananigans", xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]")
contour_levels = collect(-1:0.5:4.5)
cmap = :viridis
minstrat_s = abs.(minimum(Tz_samples_bar_average) * 1.01)
minstrat = (Tz_samples_bar_average .< minstrat) .* minstrat_s
field1 = hack_qgΨ_data
colorrange = extrema(contour_levels)
zs = (sorted_zlevels[1:end-1] + sorted_zlevels[2:end])/2
contour!(ax, latitude, zs, field1,colormap = cmap, colorrange = colorrange, levels = contour_levels, labels = true)
ax = Axis(fig[1, 2]; title = "QG Ψ: AI", xlabel = "Latitude")
field2 = hack_qgΨ_samples_mean
contour!(ax, latitude, zs, field2,colormap = cmap, colorrange = colorrange, levels = contour_levels, labels = true)
ax = Axis(fig[1, 3]; title = "QG Ψ: AI Uncertainty", xlabel = "Latitude")
field3 = hack_qgΨ_samples_std
heatmap!(ax, latitude, zs, field3, colormap = cmap, colorrange = colorrange) # , levels = contour_levels, labels = true)
save("Figures/vpTp_Tz.png", fig)


latitude = range(15, 75, length = 128)
fig = Figure() 
ax = Axis(fig[1, 1], title = "v'T' Oceananigans", xlabel = "Latitude", ylabel = "Depth")
colorrange = extrema(vpTp_data)
heatmap!(ax, latitude, sorted_zlevels, vpTp_data[1, :, :], colorrange = colorrange)
ax = Axis(fig[1, 2], title = "v'T' AI")
heatmap!(ax, latitude, sorted_zlevels, ensemble_mean_vT_samples, colorrange = colorrange)
ax = Axis(fig[2, 1], title = "Tz Data")
colorrange2 = extrema(Tz_bar)
heatmap!(ax, latitude, zs, Tz_bar[1, :, :], colorrange = colorrange2)
ax = Axis(fig[2, 2], title = "Tz AI")
heatmap!(ax, latitude, zs, Tz_samples_bar_average, colorrange = colorrange2)
ax = Axis(fig[1, 3], title = "v'T' AI Uncertainty")
heatmap!(ax, latitude, sorted_zlevels, ensemble_std_vT_samples)
ax = Axis(fig[2, 3], title = "Tz AI Uncertainty")
heatmap!(ax, latitude, zs, Tz_samples_bar_std )
save("Figures/vpTp_and_Tz.png", fig)


latitude = range(15, 75, length = 128)
fig = Figure(resolution = (1200, 600)) 
ax = Axis(fig[1, 1], title = "v'T' Oceananigans", xlabel = "Latitude", ylabel = "Depth")
colorrange = extrema(vpTp_data)
heatmap!(ax, latitude, sorted_zlevels, vpTp_data[1, :, :], colorrange = colorrange)
ax = Axis(fig[1, 2], title = "v'T' AI")
heatmap!(ax, latitude, sorted_zlevels, ensemble_mean_vT_samples, colorrange = colorrange)
ax = Axis(fig[1, 3], title = "v'T' mismatch")
maxerror = maximum(abs.(vpTp_data[1, :, :]- ensemble_mean_vT_samples))
heatmap!(ax, latitude, sorted_zlevels, vpTp_data[1, :, :]- ensemble_mean_vT_samples, colorrange = (-maxerror, maxerror), colormap = :balance)
ax = Axis(fig[2, 1], title = "Tz Data")
colorrange2 = extrema(Tz_bar)
heatmap!(ax, latitude, zs, Tz_bar[1, :, :], colorrange = colorrange2)
ax = Axis(fig[2, 2], title = "Tz AI")
heatmap!(ax, latitude, zs, Tz_samples_bar_average, colorrange = colorrange2)
ax = Axis(fig[2, 3], title = "Tz mismatch")
mterror = maximum(abs.(Tz_bar[1, :, :]-Tz_samples_bar_average))
heatmap!(ax, latitude, zs, Tz_bar[1, :, :]-Tz_samples_bar_average, colorrange = (-mterror , mterror ), colormap = :balance)
ax = Axis(fig[1, 4], title = "v'T' AI Uncertainty")
heatmap!(ax, latitude, sorted_zlevels, ensemble_std_vT_samples)
ax = Axis(fig[2, 4], title = "Tz AI Uncertainty")
heatmap!(ax, latitude, zs, Tz_samples_bar_std )
save("Figures/vpTp_and_Tz_mismatch.png", fig)