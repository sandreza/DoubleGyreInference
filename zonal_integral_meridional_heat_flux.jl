# ψ(λ, φ, t) ≡ 2 π a cos(φ) ∫∫ dλ dz' v(λ, φ, z', t)
# 2πa = 40007863, sverdrup = 1e6 m^3/s
rad_to_deg = π / 180
dλ = 60 / 128 * rad_to_deg
Lλ = 60 * rad_to_deg
# comment vbar uses the mean, technically we need to  sum, we multiply by the factor here 
ρ = 1000 
c_p = 4000
vfactor = reshape(cosd.(range(15, 75, length = 128)), (128, 1)) * 40007863 * Lλ * ρ * c_p / 10^(9) / (2π)


hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf

level_indices = [7, 5, 3, 1]
α = 2e-4
g = 9.81

# for ZLast in [32, 128]
ZLast = 128
fig = Figure(resolution = (1000, 250))
for i in eachindex(level_indices)
level_index = level_indices[i]
factor = 1
sample_tuple = return_samples_file(level_index, factor)
data_tuple = return_data_file(level_index)

μ, σ = return_scale(data_tuple)
field = data_tuple.field_2
field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
field[:, :, 4] .= field[:, :, 4] ./ (α * g) .+ 273.15
average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g) .+ 273.15

zonal_average = mean(field[1:ZLast, :, 3] .* (field[1:ZLast, :, 4] .- mean(field[1:ZLast, :, 4], dims = 1)), dims = 1)[:]
zonal_average_samples = mean(average_samples[1:ZLast, :, 3, :] .* (average_samples[1:ZLast, :, 4, :] .- mean(average_samples[1:ZLast, :, 4, :], dims = 1)), dims = 1)[1, :, :]
mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
mean_zonal_average_samples - zonal_average

mean(zonal_average)
mean(mean_zonal_average_samples)


qu = 0.99
op = 0.3
ms = 1
Nlat = size(zonal_average_samples, 1)
Nsamples = size(zonal_average_samples, 2)
latitude = range(15, 75, length = Nlat)
data_string = @sprintf("%.0f", abs(data_tuple.zlevel))
#=
ax = Axis(fig[1, i]; title = "Flux At Depth = " * data_string * " [m]", ylabel = "Latitude [ᵒ]", xlabel = "Vertical Heat Flux [K m/s]")
lines!(ax, zonal_average, latitude; color = (:blue, op), label = "Ground Truth")
scatter!(ax, zonal_average .* vfactor, latitude; color = (:blue, op), markersize = ms)
scatter!(ax, mean_zonal_average_samples .* vfactor, latitude; color = (:red, op), markersize = ms)
lines!(ax, mean_zonal_average_samples .* vfactor, latitude; color = (:red, op), label = "Generative AI")
# band(Point.(xlow, y), Point.(xhigh, y))
δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(zonal_average_samples[i, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
save("Figures/zonal_average_heat_flux_level_index_$(level_index)_factor_$(factor).png", fig)
=#

# zonal_average = mean(field[1:ZLast, :, 2] .* field[1:ZLast, :, 4], dims = 1)[:]
# zonal_average_samples = mean(average_samples[1:ZLast, :, 2, :] .* average_samples[1:ZLast, :, 4, :], dims = 1)[1, :, :]
zonal_average = mean((field[1:ZLast, :, 3] .- mean(field[1:ZLast, :, 3], dims = 1)) .* (field[1:ZLast, :, 4] .- mean(field[1:ZLast, :, 4], dims = 1)), dims = 1)[:]
zonal_average_samples = mean((average_samples[1:ZLast, :, 3, :] .- mean(average_samples[1:ZLast, :, 3, :], dims = 1) ).* (average_samples[1:ZLast, :, 4, :] .- mean(average_samples[1:ZLast, :, 4, :], dims = 1)), dims = 1)[1, :, :]

mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
mean_zonal_average_samples - zonal_average
mean(zonal_average)
mean(mean_zonal_average_samples)

ax = Axis(fig[1, i]; title = "Depth = " * data_string * " [m]", ylabel = "Latitude [ᵒ]", xlabel = "Meridional Heat Flux [GW m⁻¹]")
lines!(ax, zonal_average .* vfactor[:], latitude; color = (:blue, op), label = "Oceananigans")
scatter!(ax, zonal_average .* vfactor[:], latitude; color = (:blue, op), markersize = ms)
scatter!(ax, mean_zonal_average_samples .* vfactor[:], latitude; color = (:red, op), markersize = ms)
lines!(ax, mean_zonal_average_samples .* vfactor[:], latitude; color = (:red, op), label = "AI")
# band(Point.(xlow, y), Point.(xhigh, y))
δlower = [quantile(zonal_average_samples[i, :] .* vfactor[i], 1-qu) for i in 1:Nlat]
δupper = [quantile(zonal_average_samples[i, :] .* vfactor[i], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
if i > 1
    hideydecorations!(ax; hiding_options...)
end

if i == 4
    axislegend(ax, position = :lb)
end

end

save("Figures/zonal_average_heat_flux_depths_$(factor)_$(ZLast)_new_units.png", fig)
# end
