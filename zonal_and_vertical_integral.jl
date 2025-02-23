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


rad_to_deg = π / 180
dλ = 60 / 128 * rad_to_deg
Lλ = 60 * rad_to_deg
# comment vbar uses the mean, technically we need to  sum, we multiply by the factor here 
ρ = 1000 
c_p = 4000
vfactor = reshape(cosd.(range(15, 75, length = 128)), (1, 128,1, 1)) * 40007863 * ρ * c_p / 10^(15) / (2π)


Δz = reshape(dz, (1, 1, 15))
vT_data = sum(sorted_vlevels_data .* Δz .* sorted_Tlevel_data, dims = (1, 3))[:] * dλ .* vfactor[:]
Δz_2 = reshape(dz, (1, 1, 15, 1))
vT_samples = sum(sorted_vlevels_samples .* Δz_2 .* sorted_Tlevels_samples, dims = (1, 3)) * dλ .* vfactor
ensemble_mean_vT_samples = mean(vT_samples, dims = 4)[1, :, 1, 1] 

Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.1
fig = Figure()
ax = Axis(fig[1, 1]; title = "Zonal and Vertically Integrated Meridional Heat Flux",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
lines!(ax, vT_data[:], latitude; color = (:blue, 0.3), label = "Oceananigans")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.3), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), label = "AI")
δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
axislegend(ax, position = :rt)
save("Figures/integrated_meridional_heat_flux_ten.png", fig)

Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.01
fig = Figure( resolution = (400, 300))
# Zonal and Vertically Integrated Meridional Heat Flux
ax = Axis(fig[1, 1]; title = "",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
lines!(ax, vT_data[:], latitude; color = (:blue, 0.3), label = "Oceananigans")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.3), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), label = "AI")
δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
axislegend(ax, position = :rt)
save("Figures/integrated_meridional_heat_flux_one.png", fig)
