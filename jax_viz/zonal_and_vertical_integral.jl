using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5
α = 2e-4
g = 9.81

total_levels =Nz = 15
Lz = 1800
nsamples = 100
sorted_vlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_data  = zeros(128, 128, total_levels)
sorted_Tlevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_vlevels_samples = zeros(128, 128, total_levels, nsamples)

file_string = "velocity_uc_production_jax_samples_"
# file_string = "regular_uc_production_jax_samples_"
# file_string = "velocity_production_jax_samples_"
# file_string = "regular_production_jax_samples_"
# file_string = "production_jax_samples_"

σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevels = z_centers.(1:Nz)

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

future_year = 50
for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string)
    sorted_Tlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) /(α * g)
    sorted_Tlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) /(α * g)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string)
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
vT_data = sum(sorted_vlevels_data .* Δz .* sorted_Tlevels_data , dims = (1, 3))[:] * dλ .* vfactor[:]
Δz_2 = reshape(dz, (1, 1, 15, 1))
vT_samples = sum(sorted_vlevels_samples .* Δz_2 .* sorted_Tlevels_samples, dims = (1, 3)) * dλ .* vfactor
ensemble_mean_vT_samples = mean(vT_samples, dims = 4)[1, :, 1, 1] 
ensemble_mean_vT_std = std(vT_samples, dims = 4)[1, :, 1, 1] 

Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.9
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
save("Figures/jax_integrated_meridional_heat_flux_ten_$(future_year).png", fig)

Nlat = size(vT_samples, 2)
op = 0.3
qu = 0.99
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
save("Figures/jax_integrated_meridional_heat_flux_one_$(future_year).png", fig)

crange = (0, 30)
erange = (0, 0.085)
fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; )
f1 = mean(sorted_Tlevels_data, dims = 1)[1, :, :]
heatmap!(ax, latitude, zlevels, f1, colorrange = crange)
ax = Axis(fig[1, 2]; )
f2 = mean(sorted_Tlevels_samples, dims = (1, 4))[1, :, :, 1]
heatmap!(ax, latitude, zlevels,  f2, colorrange = crange)
ax = Axis(fig[1, 3]; )
heatmap!(ax, latitude,  zlevels, abs.(f2 - f1), colorrange = erange)
ax = Axis(fig[1, 4]; )
f3 = std(mean(sorted_Tlevels_samples, dims = 1), dims = 4)[1, :, :, 1]
heatmap!(ax, latitude,  zlevels, f3, colorrange = erange, colormap = :viridis)
ax = Axis(fig[2, 1]; )
f1 = mean(sorted_Tlevels_samples, dims = 1)[1, :, :, 1]
heatmap!(ax, latitude, zlevels, f1, colorrange = crange)
ax = Axis(fig[2, 2]; )
f1 = mean(sorted_Tlevels_samples, dims = 1)[1, :, :, 2]
heatmap!(ax, latitude, zlevels,  f1, colorrange = crange)
ax = Axis(fig[2, 3]; )
f1 = mean(sorted_Tlevels_samples, dims = 1)[1, :, :, 3]
heatmap!(ax, latitude, zlevels, f1, colorrange = crange)
ax = Axis(fig[2, 4]; )
f1 = mean(sorted_Tlevels_samples, dims = 1)[1, :, :, 4]
heatmap!(ax, latitude, zlevels,  f1, colorrange = crange)
save("Figures/jax_check_stratification_$(future_year).png", fig)

crange = (-0.01, 0.01)
erange = (0, 0.0009)
fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; )
f1 = mean(sorted_vlevels_data, dims = 1)[1, :, :]
heatmap!(ax, latitude, zlevels, f1, colorrange = crange, colormap = :balance)
ax = Axis(fig[1, 2]; )
f2 = mean(sorted_vlevels_samples, dims = (1, 4))[1, :, :, 1]
heatmap!(ax, latitude, zlevels,  f2, colorrange = crange, colormap = :balance)
ax = Axis(fig[1, 3]; )
heatmap!(ax, latitude,  zlevels, abs.(f2 - f1), colorrange = erange, colormap = :viridis)
ax = Axis(fig[1, 4]; )
f3 = std(mean(sorted_vlevels_samples, dims = 1), dims = 4)[1, :, :, 1]
heatmap!(ax, latitude,  zlevels, f3, colorrange = erange, colormap = :viridis)
ax = Axis(fig[2, 1]; )
f1 = mean(sorted_vlevels_samples, dims = 1)[1, :, :, 1]
heatmap!(ax, latitude, zlevels, f1, colorrange = crange, colormap = :balance)
ax = Axis(fig[2, 2]; )
f1 = mean(sorted_vlevels_samples, dims = 1)[1, :, :, 2]
heatmap!(ax, latitude, zlevels,  f1, colorrange = crange, colormap = :balance)
ax = Axis(fig[2, 3]; )
f1 = mean(sorted_vlevels_samples, dims = 1)[1, :, :, 3]
heatmap!(ax, latitude, zlevels, f1, colorrange = crange, colormap = :balance)
ax = Axis(fig[2, 4]; )
f1 = mean(sorted_vlevels_samples, dims = 1)[1, :, :, 4]
heatmap!(ax, latitude, zlevels,  f1, colorrange = crange, colormap = :balance)
save("Figures/jax_check_vaverage_$(future_year).png", fig)

## Check statistics 
fig = Figure()
ax = Axis(fig[1, 1]; title = "Ground Truth")
heatmap!(ax, sorted_vlevels_data[:, :, 15], colorrange = (-1, 1), colormap = :balance)
ax = Axis(fig[1, 2]; title = "AI Mean")
heatmap!(ax, mean(sorted_vlevels_samples[:, :, 15, :], dims = 3)[:, :, 1], colorrange = (-1, 1), colormap = :balance)
ax = Axis(fig[1, 3]; title = "Difference")
heatmap!(ax, mean(sorted_vlevels_samples[:, :, 15, :], dims = 3)[:, :, 1] - sorted_vlevels_data[:, :, 15], colorrange = (-0.1, 0.1), colormap = :balance)
ax = Axis(fig[1, 4]; title = "AI Uncertainty")
heatmap!(ax, std(sorted_vlevels_samples[:, :, 15, :], dims = 3)[:, :, 1], colorrange = (0, 0.1), colormap = :viridis)

level = 10
ax = Axis(fig[2, 1]; title = "Ground Truth")
heatmap!(ax, sorted_vlevels_data[:, :, level], colorrange = (-1, 1), colormap = :balance)
ax = Axis(fig[2, 2]; title = "AI Mean")
heatmap!(ax, mean(sorted_vlevels_samples[:, :, level, :], dims = 3)[:, :, 1], colorrange = (-1, 1), colormap = :balance)
ax = Axis(fig[2, 3]; title = "Difference")
heatmap!(ax, mean(sorted_vlevels_samples[:, :, level, :], dims = 3)[:, :, 1] - sorted_vlevels_data[:, :, level], colorrange = (-0.1, 0.1), colormap = :balance)
ax = Axis(fig[2, 4]; title = "AI Uncertainty")
heatmap!(ax, std(sorted_vlevels_samples[:, :, level, :], dims = 3)[:, :, 1], colorrange = (0, 0.1), colormap = :viridis)

save("Figures/jax_check_vstats_$(future_year).png", fig)