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

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)


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


# ψ(λ, φ, t) ≡ 2 π a cos(φ) ∫∫ dλ dz' v(λ, φ, z', t)
# 2πa = 40007863, sverdrup = 1e6 m^3/s
rad_to_deg = π / 180
dλ = 60 / 128 * rad_to_deg
Lλ = 60 * rad_to_deg

# comment vbar uses the mean, technically we need to  sum, we multiply by the factor here 
R_earth = 40007863 / (2π)
vfactor = reshape(cosd.(range(15, 75, length = 128)), (128, 1)) * R_earth / 1e6  * Lλ 
Δz = reshape(dz, (1, 15))

Ψᵛ_data = reverse(vfactor .* cumsum(reverse(v̄_data .* Δz, dims = 2), dims = 2), dims = 2)
Ψᵛ_samples = reverse(vfactor .* cumsum(reverse(v̄_samples .* Δz, dims = 2), dims = 2), dims = 2)
Ψᵛ_sample_members = reverse(vfactor .* cumsum(reverse(mean(sorted_vlevels_samples, dims = 1) .* reshape(Δz, (1, 1, 15, 1)), dims = 3), dims = 3)[1, :, :, :], dims = 2)
Ψᵛ_sample_std = reverse(vfactor .* std(cumsum(reverse(mean(sorted_vlevels_samples, dims = 1) .* reshape(Δz, (1, 1, 15, 1)), dims = 3), dims = 3), dims = 4)[1, :, :, 1], dims = 2)
Ψʷ_data = cumsum(w̄_data, dims = 1) * dy[1]
Ψʷ_samples = cumsum(w̄_samples, dims = 1) * dy[1]
Ψʷ_sample_std = std(cumsum(mean(sorted_wlevels_samples, dims = 1), dims = 2), dims = 4)[1, :, :, 1] * dy[1]

error_per_level_and_member = zeros(size(Ψᵛ_sample_members)[2:3]...)
for i in 1:size(error_per_level_and_member, 1)
    for j in 1:size(error_per_level_and_member, 2)
        error_per_level_and_member[i, j] = norm(Ψᵛ_data[:, i] - Ψᵛ_sample_members[:, i, j], Inf)
    end
end

ensemble_min = [argmin(error_per_level_and_member[i, :]) for i in 1:15]
Ψᵛ_sample_members_best = similar(Ψᵛ_data)
for i in 1:15
    Ψᵛ_sample_members_best[:, i] .= Ψᵛ_sample_members[:, i, ensemble_min[i]]
end
errors_Ψᵛ_2 = [norm(Ψᵛ_data - Ψᵛ_sample_members[:, :, i], Inf) for i in 1:size(Ψᵛ_sample_members, 3)]
errors_Ψᵛ_1 = norm(Ψᵛ_data - Ψᵛ_samples, Inf)
errors_Ψᵛ_best = norm(Ψᵛ_data - Ψᵛ_sample_members_best, Inf)

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

fig = Figure(resolution = (2000, 1000))
lats = range(15, 75, length = 128)
ax = Axis(fig[1, 1]; title = "Stream Function W Data", xlabel = latlabel, ylabel = depthlabel)
cr = (-maximum(abs.(Ψʷ_data)), maximum(abs.(Ψʷ_data)))
contour!(ax, lats, sorted_zlevels, Ψʷ_data, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 2]; title = "Stream Function W Samples", xlabel = latlabel, ylabel = depthlabel)
contour!(ax, lats, sorted_zlevels, Ψʷ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 3]; title = "Stream Function W Error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψʷ_data - Ψʷ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[1, 4]; title = "Stream Function W Sample Std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψʷ_sample_std, colormap = :viridis, colorrange = (0, maximum(Ψʷ_sample_std)))
ax = Axis(fig[2, 1]; title = "Stream Function V Data", xlabel = latlabel, ylabel = depthlabel)
cr = (-maximum(abs.(Ψᵛ_data)), maximum(abs.(Ψᵛ_data)))
contour!(ax, lats, sorted_zlevels, Ψᵛ_data, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 2]; title = "Stream Function V Samples", xlabel = latlabel, ylabel = depthlabel)
contour!(ax, lats, sorted_zlevels, Ψᵛ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 3]; title = "Stream Function V Error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_data - Ψᵛ_samples, colormap = :balance, colorrange = cr)
ax = Axis(fig[2, 4]; title = "Stream Function V Sample Std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_sample_std, colormap = :viridis, colorrange = (0, maximum(Ψᵛ_sample_std)))
save("Figures/moc_prototype_contour.png", fig)


fig = Figure(resolution = (2000, 375))
lats = range(15, 75, length = 128)
val = quantile(abs.(Ψᵛ_data[:] - Ψᵛ_samples[:]), 1.0) #maximum(abs.(Ψᵛ_data))
cr_error = (-val, val)
cr = extrema(Ψᵛ_data)
contour_levels = collect(-15:5:60)
ax = Axis(fig[1, 1]; title = "Model Output", xlabel = latlabel, ylabel = depthlabel)
contour!(ax, lats, sorted_zlevels, Ψᵛ_data, colormap = :viridis, colorrange = cr, levels = contour_levels,  labels = true)
ax = Axis(fig[1, 2]; title = "Generative Ensemble Average", xlabel = latlabel, ylabel = depthlabel)
contour!(ax, lats, sorted_zlevels, Ψᵛ_samples, colormap = :viridis, colorrange = cr, levels = contour_levels, labels = true)
ax = Axis(fig[1, 3]; title = "Stream Function Ensemble Error", xlabel = latlabel, ylabel = depthlabel)
hm = heatmap!(ax, lats, sorted_zlevels, Ψᵛ_data - Ψᵛ_samples, colormap = :balance, colorrange = cr_error)
Colorbar(fig[1, 4],  hm, label = "Sverdrup [10^6 m^3/s]")
ax = Axis(fig[1, 5]; title = "Generative Ensemble Std", xlabel = latlabel, ylabel = depthlabel)
hm2 = heatmap!(ax, lats, sorted_zlevels, Ψᵛ_sample_std, colormap = :viridis, colorrange = (0, cr_error[2]))
Colorbar(fig[1, 6], hm2, label = "Sverdrup [10^6 m^3/s]")
save("Figures/v_moc_prototype_contour.png", fig)

fig = Figure(resolution = (2000, 500))
lats = range(15, 75, length = 128)
cr_error = (-maximum(abs.(Ψᵛ_data)), maximum(abs.(Ψᵛ_data)))
cr = extrema(Ψᵛ_data)
ax = Axis(fig[1, 1]; title = "Model Output", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_data, colormap = :viridis, colorrange = cr)
ax = Axis(fig[1, 2]; title = "Generative Ensemble Average", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_samples, colormap = :viridis, colorrange = cr)
ax = Axis(fig[1, 3]; title = "Stream Function Error", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_data - Ψᵛ_samples, colormap = :balance, colorrange = cr_error)
ax = Axis(fig[1, 4]; title = "Generative Ensemble Std", xlabel = latlabel, ylabel = depthlabel)
heatmap!(ax, lats, sorted_zlevels, Ψᵛ_sample_std, colormap = :viridis, colorrange = (0, maximum(Ψᵛ_sample_std)))
save("Figures/v_moc_prototype_heatmap.png", fig)


#=
hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
=#

blevels = sorted_zlevels
MOC_data = Ψᵛ_data
MOC_mean = Ψᵛ_samples
MOC_samples = Ψᵛ_sample_members
MOC_std = Ψᵛ_sample_std
hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (6 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "Oceananigans")
cval = maximum(abs.(MOC_data))
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 10)
contourf!(ax, lat, blevels, MOC_data, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Mean")
contourf!(ax, lat, blevels, MOC_mean, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "Difference")
cm = contourf!(ax, lat, blevels, MOC_data -MOC_mean, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 4],  cm, label = "Sverdrup [m³ s⁻¹]")

ax = Axis(fig[2, 1], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Sample 1")
contourf!(ax, lat, blevels, MOC_samples[:, :, 1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[2, 2], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Sample 2")
contourf!(ax, lat, blevels, MOC_samples[:, :, end], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[2, 3], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Uncertainty")
cm = contourf!(ax, lat, blevels, MOC_std, colormap = :viridis, levels = std_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:white, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 4],  cm, label = "Sverdrup [m³ s⁻¹]")
save(pwd() * "/Figures/v_moc_physical.png", fig)


##
fig = Figure()
ax = Axis(fig[1, 1]; title = "v̄")
contourf!(ax,lat, sorted_zlevels, v̄_data, levels = range(-0.02, 0.02, length = 20), colormap = :balance)
ax = Axis(fig[1, 2]; title = "w̄")
contourf!(ax, lat, sorted_zlevels,  w̄_data, levels = range(-2e-5, 2e-5, length = 20), colormap = :balance)

ax = Axis(fig[2, 1]; title = "v slice mid")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[64, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[2, 2]; title = "w slice mid")
contourf!(ax, lat, sorted_zlevels,  sorted_wlevel_data[64, :, :], levels = range(-1e-4, 1e-4, length = 20), colormap = :balance)

ax = Axis(fig[3, 1]; title = "v slice beg")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[16, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[3, 2]; title = "w slice beg")
contourf!(ax, lat, sorted_zlevels,  sorted_wlevel_data[16, :, :], levels = range(-1e-4, 1e-4, length = 20), colormap = :balance)

ax = Axis(fig[4, 1]; title = "v slice end")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[96, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[4, 2]; title = "w slice end")
contourf!(ax, lat, sorted_zlevels,  sorted_wlevel_data[96, :, :], levels = range(-1e-4, 1e-4, length = 20), colormap = :balance)

ax = Axis(fig[3, 3]; title = "v slice mbeg")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[8, :, :], levels = range(-0.4, 0.4, length = 20), colormap = :balance)
ax = Axis(fig[4, 3]; title = "w slice mbeg")
contourf!(ax, lat, sorted_zlevels,  sorted_wlevel_data[8, :, :], levels = range(-2e-4, 2e-4, length = 20), colormap = :balance)

ax = Axis(fig[1, 3]; title = "v slice mend")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[120, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[2, 3]; title = "w slice mend")
contourf!(ax, lat, sorted_zlevels,  sorted_wlevel_data[120, :, :], levels = range(-1e-4, 1e-4, length = 20), colormap = :balance)

save("Figures/v_w_data.png", fig)
