using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

vlevels_data = zeros(128, 128, length(files))
wlevel_data = zeros(128, 128, length(files))
wlevels_samples = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))


levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    sample_tuple = return_samples_file(level, factor; complement = false)
    data_tuple = return_data_file(level; complement = false)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    # field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    # average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    wlevels_samples[:, :, i, :] .= average_samples[:, :, 3, :]

    vlevels_data[:, :, i] .= field[:, :, 2]
    wlevel_data[:, :, i] .= field[:, :, 3]

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
    # field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    # average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    vlevels_samples[:, :, ii, :] .= average_samples[:, :, 2, :]
    wlevels_samples[:, :, ii, :] .= average_samples[:, :, 3, :]

    vlevels_data[:, :, ii] .= field[:, :, 2]
    wlevel_data[:, :, ii] .= field[:, :, 3]
    
    zlevels[ii] = data_tuple.zlevel
end


permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_vlevels_data = vlevels_data[:, :, permuted_indices]
sorted_wlevel_data = wlevel_data[:, :, permuted_indices]
sorted_vlevels_samples = vlevels_samples[:, :, permuted_indices, :]
sorted_wlevels_samples = wlevels_samples[:, :, permuted_indices, :]

v̄_data = mean(sorted_vlevels_data, dims = 1)[1, :, :]
w̄_data = mean(sorted_wlevel_data, dims = 1)[1, :, :]
v̄_samples = mean(sorted_vlevels_samples, dims = (1, 4))[1, :, :, 1]
w̄_samples = mean(sorted_wlevels_samples, dims = (1, 4))[1, :, :, 1]
v̄_sample_std = std(mean(sorted_vlevels_samples, dims = 1), dims = 4)[1, :, :, 1]
w̄_sample_std = std(mean(sorted_wlevels_samples, dims = 1), dims = 4)[1, :, :, 1]

fig = Figure(resolution = (2000, 1000))
lats = range(15, 75, length = 128)
latlabel = "Latitude [ᵒ]"
depthlabel = "Depth [m]"
ax = Axis(fig[1, 1]; title = "w̄ data", xlabel = latlabel, ylabel = depthlabel)
cr = (-maximum(abs.(w̄_data)), maximum(abs.(w̄_data)))
heatmap!(ax, lats, sorted_zlevels, w̄_data, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 2]; title = "w̄ samples", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, w̄_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 3]; title = "w̄ error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, w̄_data - w̄_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 4]; title = "w̄ sample std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, w̄_sample_std, colormap = :viridis, colorrange = (0, maximum(w̄_sample_std)))
ax = Axis(fig[2, 1]; title = "v̄ data", xlabel = latlabel, ylabel = depthlabel)
cr = (-maximum(abs.(v̄_data)), maximum(abs.(v̄_data)))
heatmap!(ax, lats, sorted_zlevels, v̄_data, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 2]; title = "v̄ samples", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, v̄_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 3]; title = "v̄ error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, v̄_data - v̄_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 4]; title = "v̄ sample std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, v̄_sample_std, colormap = :viridis, colorrange = (0, maximum(v̄_sample_std)))
save("Figures/mean_v_w_data_samples.png", fig)


Ψᵛ_data = -cumsum(v̄_data, dims = 2)
Ψᵛ_samples = -cumsum(v̄_samples, dims = 2)
Ψᵛ_sample_std = std(cumsum(mean(sorted_vlevels_samples, dims = 1), dims = 4), dims = 4)[1, :, :, 1]
Ψʷ_data = cumsum(w̄_data, dims = 1)
Ψʷ_samples = cumsum(w̄_samples, dims = 1)
Ψʷ_sample_std = std(cumsum(mean(sorted_wlevels_samples, dims = 1), dims = 2), dims = 4)[1, :, :, 1]

fig = Figure(resolution = (2000, 1000))
lats = range(15, 75, length = 128)
ax = Axis(fig[1, 1]; title = "Stream Function W Data", xlabel = latlabel, ylabel = depthlabel)
cr = (-maximum(abs.(Ψʷ_data)), maximum(abs.(Ψʷ_data)))
heatmap!(ax, lats, sorted_zlevels, Ψʷ_data, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 2]; title = "Stream Function W Samples", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψʷ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 3]; title = "Stream Function W Error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψʷ_data - Ψʷ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 4]; title = "Stream Function W Sample Std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψʷ_sample_std, colormap = :viridis, colorrange = (0, maximum(Ψʷ_sample_std)))
ax = Axis(fig[2, 1]; title = "Stream Function V Data", xlabel = latlabel, ylabel = depthlabel)
cr = (-maximum(abs.(Ψᵛ_data)), maximum(abs.(Ψᵛ_data)))
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_data, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 2]; title = "Stream Function V Samples", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 3]; title = "Stream Function V Error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_data - Ψᵛ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 4]; title = "Stream Function V Sample Std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_sample_std, colormap = :viridis, colorrange = (0, maximum(Ψᵛ_sample_std)))
save("Figures/moc_prototype.png", fig)
