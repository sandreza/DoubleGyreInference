using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf

level_indices = [7, 5, 3, 1]

fig = Figure(resolution = (1000, 500))
for i in eachindex(level_indices)
level_index = level_indices[i]
factor = 1
sample_tuple = return_samples_file(level_index, factor)
data_tuple = return_data_file(level_index)

field = data_tuple.field_2
average_samples = sample_tuple.samples_2

zonal_average = mean(field[:, :, 3] .* field[:, :, 4], dims = 1)[:]
zonal_average_samples = mean(average_samples[:, :, 3, :] .* average_samples[:, :, 4, :], dims = 1)[1, :, :]
mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
mean_zonal_average_samples - zonal_average

mean(zonal_average)
mean(mean_zonal_average_samples)


qu = 1.0
op = 0.3
ms = 1
Nlat = size(zonal_average_samples, 1)
Nsamples = size(zonal_average_samples, 2)
latitude = range(15, 75, length = Nlat)
data_string = @sprintf("%.0f", abs(data_tuple.zlevel))
ax = Axis(fig[1, i]; title = "Flux At Depth = " * data_string * " [m]", ylabel = "Latitude [ᵒ]", xlabel = "Vertical Heat Flux [K m/s]")
lines!(ax, zonal_average, latitude; color = (:blue, op), label = "Ground Truth")
scatter!(ax, zonal_average, latitude; color = (:blue, op), markersize = ms)
scatter!(ax, mean_zonal_average_samples, latitude; color = (:red, op), markersize = ms)
lines!(ax, mean_zonal_average_samples, latitude; color = (:red, op), label = "Generative AI")
# band(Point.(xlow, y), Point.(xhigh, y))
δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(zonal_average_samples[i, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
save("Figures/zonal_average_heat_flux_level_index_$(level_index)_factor_$(factor).png", fig)

if i == 1
    axislegend(ax, position = :rc)
end

zonal_average = mean(field[:, :, 2] .* field[:, :, 4], dims = 1)[:]
zonal_average_samples = mean(average_samples[:, :, 2, :] .* average_samples[:, :, 4, :], dims = 1)[1, :, :]
mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
mean_zonal_average_samples - zonal_average
mean(zonal_average)
mean(mean_zonal_average_samples)

ax = Axis(fig[2, i]; title = "Flux At Depth = " * data_string * " [m]", ylabel = "Latitude [ᵒ]", xlabel = "Meridional Heat Flux [K m/s]")
lines!(ax, zonal_average, latitude; color = (:blue, op))
scatter!(ax, zonal_average, latitude; color = (:blue, op), markersize = ms)
scatter!(ax, mean_zonal_average_samples, latitude; color = (:red, op), markersize = ms)
lines!(ax, mean_zonal_average_samples, latitude; color = (:red, op))
# band(Point.(xlow, y), Point.(xhigh, y))
δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
δupper = [quantile(zonal_average_samples[i, :], qu) for i in 1:Nlat]
band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))

end

save("Figures/zonal_average_heat_flux_$(factor).png", fig)