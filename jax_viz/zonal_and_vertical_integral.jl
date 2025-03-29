using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5, Random
α = 2e-4
g = 9.81

total_levels =Nz = 15
Lz = 1800
nsamples = 100
sorted_vlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_data  = zeros(128, 128, total_levels)
sorted_Tlevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_vlevels_samples = zeros(128, 128, total_levels, nsamples)



future_year = 50
cg = 0 # coarse-graining level
# file_string = "less_noise_attention_velocity_uc_production_jax_samples_"
file_string = "attention_velocity_uc_production_jax_samples_"
# file_string = "attention_residual_uc_production_jax_samples_"

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


for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string, cg)
    sorted_Tlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) /(α * g)
    sorted_Tlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) /(α * g)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string, cg)
    sorted_vlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_vlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 
end

rad_to_deg = π / 180
dλ = 60 / 128 * rad_to_deg
Lλ = 60 * rad_to_deg
# comment vbar uses the mean, technically we need to  sum, we multiply by the factor here 
ρ = 1000 
c_p = 4000
vfactor = reshape(cosd.(range(15, 75, length = 128)), (1, 128,1, 1)) * 40007863 * ρ * c_p / 10^(15) / (2π)

Δz = reshape(dz, (1, 1, 15))
Δz_2 = reshape(dz, (1, 1, 15, 1))
vbar_data = mean(sum(sorted_vlevels_data .*  Δz_2, dims = 3), dims = 1) ./ sum( Δz_2) 
Tbar_data = mean(sum(sorted_Tlevels_data .*  Δz_2, dims = 3), dims = 1) ./ sum( Δz_2)
vT_data = sum((sorted_vlevels_data) .* Δz .* (sorted_Tlevels_data) , dims = (1, 3))[:] * dλ .* vfactor[:]
vT_samples = sum(sorted_vlevels_samples .* Δz_2 .* sorted_Tlevels_samples, dims = (1, 3)) * dλ .* vfactor
ensemble_mean_vT_samples = mean(vT_samples, dims = 4)[1, :, 1, 1] 
ensemble_mean_vT_std = std(vT_samples, dims = 4)[1, :, 1, 1] 
hflux = hflux .* dλ .* reshape(vfactor, (128, 1)) /(α * g)

#sum((sorted_vlevels_data .- mean(sum(sorted_vlevels_data .*  Δz_2, dims = 3), dims = 1) ./ sum( Δz_2) ) .* Δz_2, dims = (1, 3))
Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.9
fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; title = "Zonal and Vertically Integrated Meridional Heat Flux",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
lines!(ax, vT_data[:], latitude; color = (:blue, 0.3), label = "OcS")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.3), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), label = "AI")
δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
xlims!(ax, -0.2, 0.7)
axislegend(ax, position = :rt)
save("Figures/jax_integrated_meridional_heat_flux_ten_$(future_year)_$(cg).png", fig)

Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.6
fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; title = "Zonal and Vertically Integrated Meridional Heat Flux",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
lines!(ax, vT_data[:], latitude; color = (:blue, 0.3), label = "OcS")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.3), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), label = "AI")
δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
xlims!(ax, -0.2, 0.7)
axislegend(ax, position = :rt)
save("Figures/jax_integrated_meridional_heat_flux_sixty_$(future_year)_$(cg).png", fig)

#
op = 0.3
qu = 0.6
fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; title = "Zonal and Vertically Integrated Meridional Heat Flux",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
for qu in [0.6, 0.7, 0.8, 0.9]
    δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
    δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
    band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, 0.2))
end
lines!(ax, vT_data[:], latitude; color = (:blue, 0.5), label = "OcS")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.5), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.5), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.5), label = "AI")
xlims!(ax, -0.2, 0.7)
axislegend(ax, position = :rt)
save("Figures/jax_integrated_meridional_heat_flux_hack_density_$(future_year)_$(cg).png", fig)

op = 0.3
qu = 0.6
fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; title = "Zonal and Vertically Integrated Meridional Heat Flux",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
for qu in [0.6, 0.7, 0.8, 0.9]
    δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
    δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
    band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, 0.2))
end
lines!(ax, vT_data[:], latitude; color = (:blue, 0.5), label = "OcS")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.5), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.5), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.5), label = "AI")
for qu in [0.6, 0.7, 0.8, 0.9, 1.0]
    δlower = [quantile(hflux[i, 1:3400][:], 1-qu) for i in 1:Nlat]
    δupper = [quantile(hflux[i, 1:3400][:], qu) for i in 1:Nlat]
    band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:blue, 0.1))
end
xlims!(ax, -0.2, 0.7)
axislegend(ax, position = :rt)
save("Figures/jax_integrated_meridional_heat_flux_hack_density_$(future_year)_$(cg)_with_data_density.png", fig)

Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.99
fig = Figure( resolution = (400, 300))
# Zonal and Vertically Integrated Meridional Heat Flux
ax = Axis(fig[1, 1]; title = "",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
lines!(ax, vT_data[:], latitude; color = (:blue, 0.3), label = "OcS")
scatter!(ax, vT_data[:], latitude; color = (:blue, 0.3), markersize = 1)
scatter!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), markersize = 1)
lines!(ax, ensemble_mean_vT_samples, latitude; color = (:red, 0.3), label = "AI")
δlower = [quantile(vT_samples[1, i, 1, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(vT_samples[1, i, 1, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
xlims!(ax, -0.2, 0.7)
axislegend(ax, position = :rt)
save("Figures/jax_integrated_meridional_heat_flux_one_$(future_year)_$(cg).png", fig)



## Check correlation 
