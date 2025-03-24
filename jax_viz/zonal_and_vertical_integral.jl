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



future_year = 0
cg = 0 # coarse-graining level
file_string = "attention_velocity_uc_production_jax_samples_"
# file_string = "large_residual_uc_production_jax_samples_"
# file_string = "attention_velocity_uc_production_jax_samples_"
# file_string = "residual_uc_production_jax_samples_"
# file_string = "velocity_uc_production_jax_samples_"
# file_string = "regular_uc_production_jax_samples_"
# file_string = "velocity_production_jax_samples_"
# file_string = "regular_production_jax_samples_"
# file_string = "production_jax_samples_"

σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevels = z_centers.(1:Nz)

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dz = read(hfile["dz"])
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

#sum((sorted_vlevels_data .- mean(sum(sorted_vlevels_data .*  Δz_2, dims = 3), dims = 1) ./ sum( Δz_2) ) .* Δz_2, dims = (1, 3))
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

## Check correlation 
depth_correlation_v = [sum(sorted_vlevels_data[:, :, i] .* sorted_vlevels_data[:, :, j]) / (norm(sorted_vlevels_data[:, :, i]) .* norm(sorted_vlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_T = [sum(sorted_Tlevels_data[:, :, i] .* sorted_Tlevels_data[:, :, j]) / (norm(sorted_Tlevels_data[:, :, i]) .* norm(sorted_Tlevels_data[:, :, j])) for i in 1:15, j in 1:15]
##
vprime_data = sorted_vlevels_data .- sum(sorted_vlevels_data .* Δz_2[:, :, :, 1], dims = 3) / sum(Δz_2)
depth_correlation_vprime = [sum(vprime_data[:, :, i] .* vprime_data[:, :, j]) / (norm(vprime_data[:, :, i]) .* norm(vprime_data[:, :, j])) for i in 1:15, j in 1:15]
Tprime_data = sorted_Tlevels_data .- sum(sorted_Tlevels_data .* Δz_2[:, :, :, 1], dims = 3) / sum(Δz_2)
depth_correlation_Tprime = [sum(Tprime_data[:, :, i] .* Tprime_data[:, :, j]) / (norm(Tprime_data[:, :, i]) .* norm(Tprime_data[:, :, j])) for i in 1:15, j in 1:15]
##
sorted_ulevels_data = zeros(128, 128, total_levels)
sorted_wlevels_data = zeros(128, 128, total_levels)
for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :u, future_year; file_string, cg)
    sorted_ulevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    (; ground_truth, samples, mu, sigma) = jax_field(level, :w, future_year; file_string, cg)
    sorted_wlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu)
end
depth_correlation_u = [sum(sorted_ulevels_data[:, :, i] .* sorted_ulevels_data[:, :, j]) / (norm(sorted_ulevels_data[:, :, i]) .* norm(sorted_ulevels_data[:, :, j])) for i in 1:15, j in 1:15]
uprime_data = sorted_ulevels_data .- sum(sorted_ulevels_data .* Δz_2[:, :, :, 1], dims = 3) / sum(Δz_2)
depth_correlation_uprime = [sum(uprime_data[:, :, i] .* uprime_data[:, :, j]) / (norm(uprime_data[:, :, i]) .* norm(uprime_data[:, :, j])) for i in 1:15, j in 1:15]

depth_correlation_w = [sum(sorted_wlevels_data[:, :, i] .* sorted_wlevels_data[:, :, j]) / (norm(sorted_wlevels_data[:, :, i]) .* norm(sorted_wlevels_data[:, :, j])) for i in 1:15, j in 1:15]
wprime_data = sorted_wlevels_data .- sum(sorted_wlevels_data .* Δz_2[:, :, :, 1], dims = 3) / sum(Δz_2)
depth_correlation_wprime = [sum(wprime_data[:, :, i] .* wprime_data[:, :, j]) / (norm(wprime_data[:, :, i]) .* norm(wprime_data[:, :, j])) for i in 1:15, j in 1:15]

depth_correlation_uv = [sum(sorted_ulevels_data[:, :, i] .* sorted_vlevels_data[:, :, j]) / (norm(sorted_ulevels_data[:, :, i]) .* norm(sorted_vlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_uT = [sum(sorted_ulevels_data[:, :, i] .* sorted_Tlevels_data[:, :, j]) / (norm(sorted_ulevels_data[:, :, i]) .* norm(sorted_Tlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_uw = [sum(sorted_ulevels_data[:, :, i] .* sorted_wlevels_data[:, :, j]) / (norm(sorted_ulevels_data[:, :, i]) .* norm(sorted_wlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_vw = [sum(sorted_vlevels_data[:, :, i] .* sorted_wlevels_data[:, :, j]) / (norm(sorted_vlevels_data[:, :, i]) .* norm(sorted_wlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_vT = [sum(sorted_vlevels_data[:, :, i] .* sorted_Tlevels_data[:, :, j]) / (norm(sorted_vlevels_data[:, :, i]) .* norm(sorted_Tlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_wT = [sum(sorted_wlevels_data[:, :, i] .* sorted_Tlevels_data[:, :, j]) / (norm(sorted_wlevels_data[:, :, i]) .* norm(sorted_Tlevels_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_uprime_vprime = [sum(uprime_data[:, :, i] .* vprime_data[:, :, j]) / (norm(uprime_data[:, :, i]) .* norm(vprime_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_uprime_wprime = [sum(uprime_data[:, :, i] .* wprime_data[:, :, j]) / (norm(uprime_data[:, :, i]) .* norm(wprime_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_vprime_wprime = [sum(vprime_data[:, :, i] .* wprime_data[:, :, j]) / (norm(vprime_data[:, :, i]) .* norm(wprime_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_uprime_Tprime = [sum(uprime_data[:, :, i] .* Tprime_data[:, :, j]) / (norm(uprime_data[:, :, i]) .* norm(Tprime_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_vprime_Tprime = [sum(vprime_data[:, :, i] .* Tprime_data[:, :, j]) / (norm(vprime_data[:, :, i]) .* norm(Tprime_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_wprime_Tprime = [sum(wprime_data[:, :, i] .* Tprime_data[:, :, j]) / (norm(wprime_data[:, :, i]) .* norm(Tprime_data[:, :, j])) for i in 1:15, j in 1:15]
depth_correlation_v_function(ii, jj) = [sum(sorted_vlevels_data[ii, jj, i] .* sorted_vlevels_data[ii, jj, j]) / (norm(sorted_vlevels_data[ii, jj, i]) .* norm(sorted_vlevels_data[ii, jj, j])) for i in 1:15, j in 1:15]
depth_correlation_v_function(1:128, 100:128)

fig = Figure()
ax = Axis(fig[1, 1]; title = "Depth Correlation")
op = 0.5
scatterlines!(ax,  zlevels, depth_correlation_u[15, :], color = (:blue, op), label = "u")
scatterlines!(ax,  zlevels, depth_correlation_v[15, :], color = (:red, op), label = "v")
scatterlines!(ax,  zlevels, depth_correlation_w[15, :], color = (:green, op), label = "w")
scatterlines!(ax,  zlevels, depth_correlation_T[15, :], color = (:orange, op), label = "T")
ylims!(ax, (0, 1))
axislegend(ax, position = :rb)
ax = Axis(fig[1, 2]; title = "Depth Correlation")
scatterlines!(ax,  zlevels, depth_correlation_uprime[15, :], color = (:blue, op), label = "u'")
scatterlines!(ax,  zlevels, depth_correlation_vprime[15, :], color = (:red, op), label = "v'")
scatterlines!(ax,  zlevels, depth_correlation_wprime[15, :], color = (:green, op), label = "w'")
scatterlines!(ax,  zlevels, depth_correlation_Tprime[15, :], color = (:orange, op), label = "T'")
axislegend(ax, position = :rb)
ylims!(ax, (-1, 1))
save("Figures/jax_depth_correlation_$(future_year).png", fig)

##
fig = Figure(resolution = (1400, 10000))
j = 15
field = sorted_ulevels_data
for i in 1:14
    tmpi = field[:, :, i] - sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2
    println("Depth $i, ", norm(tmpi), " ", norm(field[:, :, i]))
    ax = Axis(fig[i, 1]; title = "v")
    a = quantile(field[:, :, i][:], 0.9)
    heatmap!(ax, field[:, :, i], colorrange = (-a, a), colormap = :balance)
    ax = Axis(fig[i, 2]; title = "v'")
    heatmap!(ax, tmpi, colorrange = (-a, a), colormap = :balance)
    ax = Axis(fig[i, 3]; title = "vector direction")
    heatmap!(ax, sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2, colorrange = (-a, a), colormap = :balance)
end
save("Figures/jax_u_$(future_year).png", fig)

fig = Figure(resolution = (1400, 10000))
j = 15
field = sorted_vlevels_data
for i in 1:14
    tmpi = field[:, :, i] - sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2
    println("Depth $i, ", norm(tmpi), " ", norm(field[:, :, i]))
    ax = Axis(fig[i, 1]; title = "v")
    a = quantile(field[:, :, i][:], 0.9)
    heatmap!(ax, field[:, :, i], colorrange = (-a, a), colormap = :balance)
    ax = Axis(fig[i, 2]; title = "v'")
    heatmap!(ax, tmpi, colorrange = (-a, a), colormap = :balance)
    ax = Axis(fig[i, 3]; title = "vector direction")
    heatmap!(ax, sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2, colorrange = (-a, a), colormap = :balance)
end
save("Figures/jax_v_$(future_year).png", fig)

fig = Figure(resolution = (1400, 10000))
j = 15
field = sorted_Tlevels_data
for i in 1:14
    tmpi = field[:, :, i] - sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2
    println("Depth $i, ", norm(tmpi), " ", norm(field[:, :, i]))
    ax = Axis(fig[i, 1]; title = "v")
    a = quantile(field[:, :, i][:], 0.9)
    heatmap!(ax, field[:, :, i], colorrange = (0, a), colormap = :thermometer)
    ax = Axis(fig[i, 2]; title = "v'")
    heatmap!(ax, tmpi, colorrange = (0, a), colormap = :thermometer)
    ax = Axis(fig[i, 3]; title = "vector direction")
    heatmap!(ax, sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2, colorrange = (0, a), colormap = :thermometer)
end
save("Figures/jax_T_$(future_year).png", fig)

fig = Figure(resolution = (1400, 10000))
j = 15
field = sorted_wlevels_data
for i in 1:14
    tmpi = field[:, :, i] - sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2
    println("Depth $i, ", norm(tmpi), " ", norm(field[:, :, i]))
    ax = Axis(fig[i, 1]; title = "v")
    a = quantile(field[:, :, i][:], 0.9)
    heatmap!(ax, field[:, :, i], colorrange = (-a, a), colormap = :balance)
    ax = Axis(fig[i, 2]; title = "v'")
    heatmap!(ax, tmpi, colorrange = (-a, a), colormap = :balance)
    ax = Axis(fig[i, 3]; title = "vector direction")
    heatmap!(ax, sum(field[:, :, i] .* field[:, :, j]) .* field[:, :, j] / norm(field[:, :, j])^2, colorrange = (-a, a), colormap = :balance)
end
save("Figures/jax_w_$(future_year).png", fig)

##
sorted_wlevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_ulevels_samples = zeros(128, 128, total_levels, nsamples)
for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :u, future_year; file_string, cg)
    sorted_ulevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 
    (; ground_truth, samples, mu, sigma) = jax_field(level, :w, future_year; file_string, cg)
    sorted_wlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu)
end

field_correlation(field1, field2) = [sum(field1[:, :, j, :] .* field2[:, :, j, :], dims = (1, 2)) ./ sqrt.(sum(field1[:, :, j, :] .^2, dims = (1, 2)) .* sum(field2[:, :, j, :] .^2, dims = (1, 2))) for j in 1:15]
field_correlation_s(field1, field2) = [sum(field1[:, :, 15, :] .* field2[:, :, j, :], dims = (1, 2)) ./ sqrt.(sum(field1[:, :, 15, :] .^2, dims = (1, 2)) .* sum(field2[:, :, j, :] .^2, dims = (1, 2))) for j in 1:15]

field1 = sorted_vlevels_samples
field2 = reshape(sorted_vlevels_data, (128, 128, 15, 1))
fcv = field_correlation(field1, field2)
fcv_s = field_correlation_s(field1, field1)

field1 = sorted_Tlevels_samples
field2 = reshape(sorted_Tlevels_data, (128, 128, 15, 1))
fct = field_correlation(field1, field2)
fct_s = field_correlation_s(field1, field1)

field1 = sorted_ulevels_samples
field2 = reshape(sorted_ulevels_data, (128, 128, 15, 1))
fcu = field_correlation(field1, field2)
fcu_s = field_correlation_s(field1, field1)

field1 = sorted_wlevels_samples
field2 = reshape(sorted_wlevels_data, (128, 128, 15, 1))
fcw = field_correlation(field1, field2)
fcw_s = field_correlation_s(field1, field1)

meancorrelationsu = [mean(fcu[i]) for i in 1:15]
q9correlationsu = [quantile(fcu[i][:], 0.9) for i in 1:15]
q1correlationsu = [quantile(fcu[i][:], 0.1) for i in 1:15]

meancorrelationsv = [mean(fcv[i]) for i in 1:15]
q9correlationsv = [quantile(fcv[i][:], 0.9) for i in 1:15]
q1correlationsv = [quantile(fcv[i][:], 0.1) for i in 1:15]

meancorrelationsw = [mean(fcw[i]) for i in 1:15]
q9correlationsw = [quantile(fcw[i][:], 0.9) for i in 1:15]
q1correlationsw = [quantile(fcw[i][:], 0.1) for i in 1:15]

meancorrelationst = [mean(fct[i]) for i in 1:15]
q9correlationst = [quantile(fct[i][:], 0.9) for i in 1:15]
q1correlationst = [quantile(fct[i][:], 0.1) for i in 1:15]

meancorrelationsu_s = [mean(fcu_s[i]) for i in 1:15]
q9correlationsu_s = [quantile(fcu_s[i][:], 0.9) for i in 1:15]
q1correlationsu_s = [quantile(fcu_s[i][:], 0.1) for i in 1:15]

meancorrelationsv_s = [mean(fcv_s[i]) for i in 1:15]
q9correlationsv_s = [quantile(fcv_s[i][:], 0.9) for i in 1:15]
q1correlationsv_s = [quantile(fcv_s[i][:], 0.1) for i in 1:15]

meancorrelationsw_s = [mean(fcw_s[i]) for i in 1:15]
q9correlationsw_s = [quantile(fcw_s[i][:], 0.9) for i in 1:15]
q1correlationsw_s = [quantile(fcw_s[i][:], 0.1) for i in 1:15]

meancorrelationst_s = [mean(fct_s[i]) for i in 1:15]
q9correlationst_s = [quantile(fct_s[i][:], 0.9) for i in 1:15]
q1correlationst_s = [quantile(fct_s[i][:], 0.1) for i in 1:15]

fig = Figure()
ax = Axis(fig[1, 1]; title = "Depth Correlation", xlabel = "Correlation", ylabel = "Depth [m]")
op = 0.5
op2 = 0.1
# data
sc1 = scatterlines!(ax, depth_correlation_u[15, :], zlevels, color = (:blue, op), label = "u")
sc2 = scatterlines!(ax, depth_correlation_v[15, :], zlevels, color = (:red, op), label = "v")
sc3 = scatterlines!(ax, depth_correlation_w[15, :], zlevels, color = (:green, op), label = "w ")
sc4 = scatterlines!(ax, depth_correlation_T[15, :], zlevels, color = (:orange, op), label = "T ")
# ["One", "Two"], 
axislegend(ax, [sc1, sc2, sc3, sc4], ["u", "v", "w", "T"], "Surface Field", position = :lt, orientation = :horizontal)
# nn 
scu = scatterlines!(ax, meancorrelationsu, zlevels, color = (:blue, op), label = "u with AI", marker = :star6)
band!(ax, Point.(q1correlationsu, zlevels), Point.(q9correlationsu, zlevels); color = (:blue, op2))
scv = scatterlines!(ax, meancorrelationsv, zlevels, color = (:red, op), label = "v with AI", marker = :star6)
band!(ax, Point.(q1correlationsv, zlevels), Point.(q9correlationsv, zlevels); color = (:red, op2))
scw = scatterlines!(ax, meancorrelationsw, zlevels, color = (:green, op), label = "w with AI", marker = :star6)
band!(ax, Point.(q1correlationsw, zlevels), Point.(q9correlationsw, zlevels); color = (:green, op2))
scT = scatterlines!(ax, meancorrelationst, zlevels, color = (:orange, op), label = "T with AI", marker = :star6)
band!(ax, Point.(q1correlationst, zlevels), Point.(q9correlationst, zlevels); color = (:orange, op2))

axislegend(ax, [scu, scv, scw, scT], ["u", "v", "w", "T"], "AI SSH", position = :lc, orientation = :horizontal)
xlims!(ax, (0.0, 1.1))

save("Figures/jax_depth_correlation_$(future_year)_and_ai.png", fig)


factor = 480
fig = Figure(resolution = (2*factor, 1*factor))
ax = Axis(fig[1, 1]; title = "Data Correlation with Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
op = 0.5
op2 = 0.1
# data
sc1 = scatterlines!(ax, depth_correlation_u[15, :], zlevels, color = (:blue, op), label = "u")
sc2 = scatterlines!(ax, depth_correlation_v[15, :], zlevels, color = (:red, op), label = "v")
sc3 = scatterlines!(ax, depth_correlation_w[15, :], zlevels, color = (:green, op), label = "w ")
sc4 = scatterlines!(ax, depth_correlation_T[15, :], zlevels, color = (:orange, op), label = "T ")
# ["One", "Two"], 
axislegend(ax, [sc1, sc2, sc3, sc4], ["u", "v", "w", "T"], "Data", position = :lt, orientation = :horizontal)
xlims!(ax, (0.3, 1.05))
# nn 
ax = Axis(fig[1, 2]; title = "AI Correlation with Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
scu = scatterlines!(ax, meancorrelationsu_s, zlevels, color = (:blue, op), label = "u with AI", marker = :star6)
band!(ax, Point.(q1correlationsu_s, zlevels), Point.(q9correlationsu_s, zlevels); color = (:blue, op2))
scv = scatterlines!(ax, meancorrelationsv_s, zlevels, color = (:red, op), label = "v with AI", marker = :star6)
band!(ax, Point.(q1correlationsv_s, zlevels), Point.(q9correlationsv_s, zlevels); color = (:red, op2))
scw = scatterlines!(ax, meancorrelationsw_s, zlevels, color = (:green, op), label = "w with AI", marker = :star6)
band!(ax, Point.(q1correlationsw_s, zlevels), Point.(q9correlationsw_s, zlevels); color = (:green, op2))
scT = scatterlines!(ax, meancorrelationst_s, zlevels, color = (:orange, op), label = "T with AI", marker = :star6)
band!(ax, Point.(q1correlationst_s, zlevels), Point.(q9correlationst_s, zlevels); color = (:orange, op2))

axislegend(ax, [scu, scv, scw, scT], ["u", "v", "w", "T"], "AI", position = :lt, orientation = :horizontal)
xlims!(ax, (0.3, 1.05))

save("Figures/jax_depth_correlation_$(future_year)_and_ai_check.png", fig)

factor = 400
fig = Figure(resolution = (3*factor, 1*factor))
ax = Axis(fig[1, 1]; title = "Data Correlation with Data Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
op = 0.5
op2 = 0.1
# data
sc1 = scatterlines!(ax, depth_correlation_u[15, :], zlevels, color = (:blue, op), label = "u")
sc2 = scatterlines!(ax, depth_correlation_v[15, :], zlevels, color = (:red, op), label = "v")
sc3 = scatterlines!(ax, depth_correlation_w[15, :], zlevels, color = (:green, op), label = "w ")
sc4 = scatterlines!(ax, depth_correlation_T[15, :], zlevels, color = (:orange, op), label = "T ")
# ["One", "Two"], 
axislegend(ax, [sc1, sc2, sc3, sc4], ["u", "v", "w", "T"], position = :lt, orientation = :horizontal)
xlims!(ax, (0.3, 1.05))
# nn 
ax = Axis(fig[1, 2]; title = "AI Correlation with AI Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
scu = scatterlines!(ax, meancorrelationsu_s, zlevels, color = (:blue, op), label = "u with AI")
band!(ax, Point.(q1correlationsu_s, zlevels), Point.(q9correlationsu_s, zlevels); color = (:blue, op2))
scv = scatterlines!(ax, meancorrelationsv_s, zlevels, color = (:red, op), label = "v with AI")
band!(ax, Point.(q1correlationsv_s, zlevels), Point.(q9correlationsv_s, zlevels); color = (:red, op2))
scw = scatterlines!(ax, meancorrelationsw_s, zlevels, color = (:green, op), label = "w with AI")
band!(ax, Point.(q1correlationsw_s, zlevels), Point.(q9correlationsw_s, zlevels); color = (:green, op2))
scT = scatterlines!(ax, meancorrelationst_s, zlevels, color = (:orange, op), label = "T with AI")
band!(ax, Point.(q1correlationst_s, zlevels), Point.(q9correlationst_s, zlevels); color = (:orange, op2))

# axislegend(ax, [scu, scv, scw, scT], ["u", "v", "w", "T"],  position = :lt, orientation = :horizontal)
xlims!(ax, (0.3, 1.05))

ax = Axis(fig[1, 3]; title = "AI Correlation with Data", xlabel = "Correlation", ylabel = "Depth [m]")
scu = scatterlines!(ax, meancorrelationsu, zlevels, color = (:blue, op), label = "u with AI")
band!(ax, Point.(q1correlationsu, zlevels), Point.(q9correlationsu, zlevels); color = (:blue, op2))
scv = scatterlines!(ax, meancorrelationsv, zlevels, color = (:red, op), label = "v with AI")
band!(ax, Point.(q1correlationsv, zlevels), Point.(q9correlationsv, zlevels); color = (:red, op2))
scw = scatterlines!(ax, meancorrelationsw, zlevels, color = (:green, op), label = "w with AI")
band!(ax, Point.(q1correlationsw, zlevels), Point.(q9correlationsw, zlevels); color = (:green, op2))
scT = scatterlines!(ax, meancorrelationst, zlevels, color = (:orange, op), label = "T with AI")
band!(ax, Point.(q1correlationst, zlevels), Point.(q9correlationst, zlevels); color = (:orange, op2))
# axislegend(ax, [scu, scv, scw, scT], ["u", "v", "w", "T"],  position = :lt, orientation = :horizontal)
save("Figures/jax_depth_correlation_$(future_year)_and_ai_check_more.png", fig)