using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf

level_indices = [7, 5, 3, 1]
α = 2e-4
g = 9.81

convert_units = true
density       = 1025
heat_capacity = 3997

ticks(t) = (t, [L"%$i" for i in t])

xticksV = (ticks([-200, -100, 0, 100, 200]), 
           ticks([-350, -250, -150, -50, 50]), 
           ticks([-150, -100, -50, 0, 50]),
           ticks([-50, 0, 50]))

xticksH = (ticks([-0.25, 0, 0.25, 0.5, 0.75]), 
           ticks([-0.75, -0.5, -0.25, 0, 0.25]), 
           ticks([-0.2, -0.1, 0, 0.1]),
           ticks([-0.02, -0.01, 0, 0.01, 0.02]))

qu = 0.99
op = 0.4
ms = 1
geometric_factor = cosd.(range(15, 75, length = 128))

for XLast in [32, 128]
    fig = Figure(resolution = (1000, 500))

    for i in eachindex(level_indices)
    
        level_index = level_indices[i]
        factor = 1
        sample_tuple = return_samples_file(level_index, factor)
        data_tuple = return_data_file(level_index)

        μ, σ = return_scale(data_tuple)
        field = data_tuple.field_2
        field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
        field[:, :, 4]   .= field[:, :, 4] ./ (α * g)

        average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
        average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

        zonal_mean_flux = (mean(field[1:XLast, :, 3], dims=1) .* mean(field[1:XLast, :, 4], dims=1))[:]
        zonal_average = mean(field[1:XLast, :, 3] .* field[1:XLast, :, 4], dims = 1)[:] .- zonal_mean_flux
        zonal_mean_flux_samples = (mean(average_samples[1:XLast, :, 3, :], dims=1) .* mean(average_samples[1:XLast, :, 4, :], dims = 1))[1, :, :]
        zonal_average_samples = mean(average_samples[1:XLast, :, 3, :] .* average_samples[1:XLast, :, 4, :], dims = 1)[1, :, :] .- zonal_mean_flux_samples
        mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]

        if convert_units
            zonal_average .*= density * heat_capacity .* geometric_factor 
            zonal_average_samples .*= density * heat_capacity .* geometric_factor 
            mean_zonal_average_samples .*= density * heat_capacity .* geometric_factor 
        end

        Nlat     = size(zonal_average_samples, 1)
        Nsamples = size(zonal_average_samples, 2)
        latitude = range(15, 75, length = Nlat)

        data_string = @sprintf("%.0f", abs(data_tuple.zlevel))

        Label(fig[0, i], L"\text{Depth }%$(data_string)\text{ [m]}"; tellwidth=false)

        if i == 1
            ax = Axis(fig[1, i];  ylabel = L"\text{Latitude [}^\circ\text{]}", 
                                  xlabel = L"\text{Vertical heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ticks([20, 40, 60]))
                                  # xticks = xticksV[i])
        else
            ax = Axis(fig[1, i];  ylabel = "", 
                                  xlabel = L"\text{Vertical heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]))
                                  # xticks = xticksV[i])
        end

        lines!(ax, zonal_average,              latitude; color = (:blue, op), label = L"\text{Ground Truth}", linewidth=2)
        lines!(ax, mean_zonal_average_samples, latitude; color = (:red, op),  label = L"\text{Generative AI}")
        
        δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
        δupper = [quantile(zonal_average_samples[i, :], qu)   for i in 1:Nlat]

        band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))

        if i == 2
            axislegend(ax, position = :lt)
        end

        zonal_mean_flux = (mean(field[1:XLast, :, 2], dims=1) .* mean(field[1:XLast, :, 4], dims=1))[:]
        zonal_average = mean(field[1:XLast, :, 2] .* field[1:XLast, :, 4], dims = 1)[:] .- zonal_mean_flux
        zonal_mean_flux_samples = (mean(average_samples[1:XLast, :, 2, :], dims=1) .* mean(average_samples[1:XLast, :, 4, :], dims = 1))[1, :, :]
        zonal_average_samples = mean(average_samples[1:XLast, :, 2, :] .* average_samples[1:XLast, :, 4, :], dims = 1)[1, :, :] .- zonal_mean_flux_samples
        mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
        
        if convert_units
            zonal_average .*= density * heat_capacity .* geometric_factor 
            zonal_average_samples .*= density * heat_capacity .* geometric_factor 
            mean_zonal_average_samples .*= density * heat_capacity .* geometric_factor 
        end

        if i == 1
            ax = Axis(fig[2, i];  ylabel = L"\text{Latitude [}^\circ\text{]}", 
                                  xlabel = L"\text{Meridional heat flux [MWm}^{-2}\text{]}", 
                                  yticks = ticks([20, 40, 60]),
                                  xticks = xticksH[i])
        else
            ax = Axis(fig[2, i];  ylabel = "", 
                                  xlabel = L"\text{Meridional heat flux [MWm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]),
                                  xticks = xticksH[i])
        end

        lines!(ax, zonal_average ./ 1e6,              latitude; color = (:blue, op), linewidth=2)
        lines!(ax, mean_zonal_average_samples ./ 1e6, latitude; color = (:red, op))

        δlower = [quantile((zonal_average_samples[i, :]) ./ 1e6, 1-qu) for i in 1:Nlat]
        δupper = [quantile((zonal_average_samples[i, :]) ./ 1e6, qu) for i in 1:Nlat]
        band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
    end

    save("Figures/zonal_average_heat_flux_all_depths_$(XLast).png", fig)
    save("Figures/zonal_average_heat_flux_all_depths_$(XLast).eps", fig)
end

##

factors = [1, 4, 16, 64]
for XLast in [32, 128]
    fig = Figure(resolution = (1000, 500))

    for i in eachindex(factors)
    
        level_index = 3
        factor = factors[i]
        sample_tuple = return_samples_file(level_index, factor)
        data_tuple = return_data_file(level_index)

        μ, σ = return_scale(data_tuple)
        field = data_tuple.field_2
        field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
        field[:, :, 4]   .= field[:, :, 4] ./ (α * g)

        average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
        average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

        zonal_mean_flux = (mean(field[1:XLast, :, 3], dims=1) .* mean(field[1:XLast, :, 4], dims=1))[:]
        zonal_average = mean(field[1:XLast, :, 3] .* field[1:XLast, :, 4], dims = 1)[:] .- zonal_mean_flux
        zonal_mean_flux_samples = (mean(average_samples[1:XLast, :, 3, :], dims=1) .* mean(average_samples[1:XLast, :, 4, :], dims = 1))[1, :, :]
        zonal_average_samples = mean(average_samples[1:XLast, :, 3, :] .* average_samples[1:XLast, :, 4, :], dims = 1)[1, :, :] .- zonal_mean_flux_samples
        mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]

        if convert_units
            zonal_average .*= density * heat_capacity .* geometric_factor 
            zonal_average_samples .*= density * heat_capacity .* geometric_factor 
            mean_zonal_average_samples .*= density * heat_capacity .* geometric_factor 
        end

        Nlat     = size(zonal_average_samples, 1)
        Nsamples = size(zonal_average_samples, 2)
        latitude = range(15, 75, length = Nlat)

        data_string = @sprintf("%.0f", abs(factor))

        Label(fig[0, i], L"\text{Coarse Graining Factor }%$(data_string)\text{ }"; tellwidth=false)

        if i == 1
            ax = Axis(fig[1, i];  ylabel = L"\text{Latitude [}^\circ\text{]}", 
                                  xlabel = L"\text{Vertical heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ticks([20, 40, 60]),
                                  xticks = xticksV[i])
        else
            ax = Axis(fig[1, i];  ylabel = "", 
                                  xlabel = L"\text{Vertical heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]),
                                  xticks = xticksV[i])
        end

        lines!(ax, zonal_average,              latitude; color = (:blue, op), label = L"\text{Ground Truth}", linewidth=2)
        lines!(ax, mean_zonal_average_samples, latitude; color = (:red, op),  label = L"\text{Generative AI}")
        
        δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
        δupper = [quantile(zonal_average_samples[i, :], qu)   for i in 1:Nlat]

        band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))

        if i == 1
            axislegend(ax, position = :lt)
        end

        zonal_mean_flux = (mean(field[1:XLast, :, 2], dims=1) .* mean(field[1:XLast, :, 4], dims=1))[:]
        zonal_average = mean(field[1:XLast, :, 2] .* field[1:XLast, :, 4], dims = 1)[:] .- zonal_mean_flux
        zonal_mean_flux_samples = (mean(average_samples[1:XLast, :, 2, :], dims=1) .* mean(average_samples[1:XLast, :, 4, :], dims = 1))[1, :, :]
        zonal_average_samples = mean(average_samples[1:XLast, :, 2, :] .* average_samples[1:XLast, :, 4, :], dims = 1)[1, :, :] .- zonal_mean_flux_samples
        mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
        
        if convert_units
            zonal_average .*= density * heat_capacity .* geometric_factor 
            zonal_average_samples .*= density * heat_capacity .* geometric_factor 
            mean_zonal_average_samples .*= density * heat_capacity .* geometric_factor 
        end

        if i == 1
            ax = Axis(fig[2, i];  ylabel = L"\text{Latitude [}^\circ\text{]}", 
                                  xlabel = L"\text{Meridional heat flux [MWm}^{-2}\text{]}", 
                                  yticks = ticks([20, 40, 60]),
                                  xticks = xticksH[i])
        else
            ax = Axis(fig[2, i];  ylabel = "", 
                                  xlabel = L"\text{Meridional heat flux [MWm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]),
                                  xticks = xticksH[i])
        end

        lines!(ax, zonal_average ./ 1e6,              latitude; color = (:blue, op), linewidth=2)
        lines!(ax, mean_zonal_average_samples ./ 1e6, latitude; color = (:red, op))

        δlower = [quantile((zonal_average_samples[i, :]) ./ 1e6, 1-qu) for i in 1:Nlat]
        δupper = [quantile((zonal_average_samples[i, :]) ./ 1e6, qu) for i in 1:Nlat]
        band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
    end

    save("Figures/zonal_average_heat_flux_fixed_depth_coarse_graining_$(XLast)_coarse_graining.png", fig)
    save("Figures/zonal_average_heat_flux_fixed_depth_coarse_graining_$(XLast).eps", fig)
end

