using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf

level_indices = [7, 5, 3, 1]
α = 2e-4
g = 9.81

convert_units = true
density       = 1025
heat_capacity = 3997

ticks(t) = (t, [L"%$i" for i in t])

xticksV = (ticks([-2000, -1000, 0, 1000, 2000]), 
          ticks([-1000, -500, 0, 500, 1000]), 
          ticks([-500, -250, 0, 250, 500]),
          ticks([-50, 150, 350, 550]))

xticksH = (ticks([0, 2.5, 5]), 
          ticks([0, 1, 2, 3]), 
          ticks([0, 0.3, 0.6]),
          ticks([-0.1, 0, 0.1]))

qu = 0.995
op = 0.4
ms = 1

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

        zonal_average = mean(field[1:XLast, :, 3] .* field[1:XLast, :, 4], dims = 1)[:]
        zonal_average_samples = mean(average_samples[1:XLast, :, 3, :] .* average_samples[1:XLast, :, 4, :], dims = 1)[1, :, :]
        mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]
        
        if convert_units
            zonal_average .*= density * heat_capacity
            zonal_average_samples .*= density * heat_capacity
            mean_zonal_average_samples .*= density * heat_capacity
        end

        Nlat     = size(zonal_average_samples, 1)
        Nsamples = size(zonal_average_samples, 2)
        latitude = range(15, 75, length = Nlat)

        data_string = @sprintf("%.0f", abs(data_tuple.zlevel))

        Label(fig[0, i], L"\text{Depth }%$(data_string)\text{ [m]}"; tellwidth=false)

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

        if i == 4
            axislegend(ax, position = :rc)
        end

        zonal_average = mean(field[1:XLast, :, 2] .* field[1:XLast, :, 4], dims = 1)[:]
        zonal_average_samples = mean(average_samples[1:XLast, :, 2, :] .* average_samples[1:XLast, :, 4, :], dims = 1)[1, :, :]
        mean_zonal_average_samples = mean(zonal_average_samples, dims = 2)[:]

        if convert_units
            zonal_average .*= density * heat_capacity
            zonal_average_samples .*= density * heat_capacity
            mean_zonal_average_samples .*= density * heat_capacity
        end

        if i == 1
            ax = Axis(fig[2, i];  ylabel = L"\text{Latitude [}^\circ\text{]}", 
                                  xlabel = L"\text{Meridional heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ticks([20, 40, 60]),
                                  xticks = xticksH[i])
        else
            ax = Axis(fig[2, i];  ylabel = "", 
                                  xlabel = L"\text{Meridional heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]),
                                  xticks = xticksH[i])
        end

        lines!(ax, zonal_average ./ 1e6,              latitude; color = (:blue, op), linewidth=2)
        lines!(ax, mean_zonal_average_samples ./ 1e6, latitude; color = (:red, op))

        δlower = [quantile(zonal_average_samples[i, :] ./ 1e6, 1-qu) for i in 1:Nlat]
        δupper = [quantile(zonal_average_samples[i, :] ./ 1e6, qu) for i in 1:Nlat]
        band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
    end

    save("Figures/zonal_average_heat_flux_depths_$(factor)_$(XLast).png", fig)
end

##
