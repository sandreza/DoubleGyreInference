using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf

future_year = 50
file_string = "attention_velocity_uc_production_jax_samples_"
level_indices = [14, 12, 10, 3]

α = 2e-4
g = 9.81

Nz = 15
Lz = 1800
σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevel = z_centers.(1:Nz)


convert_units = true
density       = 1025
heat_capacity = 3997

# XLast = 32, gives the eddy contribution of the gyre

ticks(t) = (t, [L"%$i" for i in t])

# optimized for 128
xticksV = (ticks([-20, -10, 0, 10, 20]), 
           ticks([-50, -25, 0, 25]), 
           ticks([-30, -20, -10, 0, 10]),
           ticks([-10, -5,  0]))

xticksH = (ticks([0, 0.1, 0.2, 0.3]), 
           ticks([-0.1, -0.05, 0, 0.05]), 
           ticks([-0.02, -0.01, 0, 0.01] .* 2),
           ticks([-0.001, 0, 0.001, 0.002]))

op = 0.2
op2 = 0.4
ms = 1
geometric_factor = cosd.(range(15, 75, length = 128))

for XLast in [128]
    fig = Figure(resolution = (1000, 500))
    for i in eachindex(level_indices)
        factor = 1
        level_index = level_indices[i]
        field = zeros(128, 128, 4);
        average_samples = zeros(128, 128, 4, 100);
        for (field_index, field_symbol) in enumerate([:u, :v, :w, :b])
            (; ground_truth, samples, mu, sigma) = jax_field(level_index, field_symbol, future_year; file_string, cg=factor-1)
            if field_index == 4
                field[:, :, field_index] .= (ground_truth .* sigma .+ mu)/(α * g)
                average_samples[:, :, field_index, :] .= (samples .* sigma .+ mu)/(α * g)
            else
                field[:, :, field_index] .= (ground_truth .* sigma .+ mu)
                average_samples[:, :, field_index, :] .= (samples .* sigma .+ mu) 
            end
        end

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

        data_string = @sprintf("%.0f", abs(zlevel[level_index]))

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

        lines!(ax, mean_zonal_average_samples, latitude; color = (:red, op),  label = L"\text{Generative AI}")
        
        for qu in [0.6, 0.7, 0.8, 0.9, 1.0]
            δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
            δupper = [quantile(zonal_average_samples[i, :], qu) for i in 1:Nlat]
            band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
        end
        lines!(ax, zonal_average,              latitude; color = (:blue, op2), label = L"\text{Ground Truth}", linewidth=2)
        if i == 4
            axislegend(ax, position = :lb)
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


        lines!(ax, mean_zonal_average_samples ./ 1e6, latitude; color = (:red, op))
        for qu in [0.6, 0.7, 0.8, 0.9, 1.0]
            δlower = [quantile(zonal_average_samples[i, :], 1-qu) ./ 1e6 for i in 1:Nlat]
            δupper = [quantile(zonal_average_samples[i, :], qu) ./ 1e6 for i in 1:Nlat]
            band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
        end
        lines!(ax, zonal_average ./ 1e6,              latitude; color = (:blue, op2), linewidth=2)
    end

    save("Figures/zonal_average_heat_flux_all_depths_$(XLast).png", fig)
    save("Figures/zonal_average_heat_flux_all_depths_$(XLast).eps", fig)
end


vrange = [-40, -30, -20, -10, 0, 10, 20]
xticksV = (ticks(vrange[2:end-1]), 
           ticks(vrange[2:end-1]), 
           ticks(vrange[2:end-1]), 
           ticks(vrange[2:end-1]))

hrange = [-0.04, -0.02, -0.01, 0, 0.01, 0.02] .* 3
xticksH = (ticks(hrange[2:end-1]), 
           ticks(hrange[2:end-1]), 
           ticks(hrange[2:end-1]), 
           ticks(hrange[2:end-1]))

factors = [1, 5, 6, 8]
for XLast in [128]
    fig = Figure(resolution = (1000, 500))

    for i in eachindex(factors)
        level_index = 10
        factor = factors[i]
        field = zeros(128, 128, 4)
        average_samples = zeros(128, 128, 4, 100)
        for (field_index, field_symbol) in enumerate([:u, :v, :w, :b])
            (; ground_truth, samples, mu, sigma) = jax_field(level_index, field_symbol, future_year; file_string, cg=factor-1)
            if field_index == 4
                field[:, :, field_index] .= (ground_truth .* sigma .+ mu)/(α * g)
                average_samples[:, :, field_index, :] .= (samples .* sigma .+ mu)/(α * g)
            else
                field[:, :, field_index] .= (ground_truth .* sigma .+ mu)
                average_samples[:, :, field_index, :] .= (samples .* sigma .+ mu) 
            end
        end

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

        data_string = @sprintf("%.0f", 2^abs(factor-1))

        Label(fig[0, i], L"\text{Coarse Graining Factor }%$(data_string)\text{ }"; tellwidth=false)

        if i == 1
            ax = Axis(fig[1, i];  ylabel = L"\text{Latitude [}^\circ\text{]}", 
                                  xlabel = L"\text{Vertical heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ticks([20, 40, 60]),
                                  xticks = xticksV[i])
                                  xlims!(ax, extrema(vrange)...)
        else
            ax = Axis(fig[1, i];  ylabel = "", 
                                  xlabel = L"\text{Vertical heat flux [Wm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]),
                                  xticks = xticksV[i])
                                  xlims!(ax, extrema(vrange)...)
        end


        lines!(ax, mean_zonal_average_samples, latitude; color = (:red, op),  label = L"\text{Generative AI}")
        
        for qu in [0.6, 0.7, 0.8, 0.9, 1.0]
            δlower = [quantile(zonal_average_samples[i, :], 1-qu) for i in 1:Nlat]
            δupper = [quantile(zonal_average_samples[i, :], qu) for i in 1:Nlat]
            band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
        end
        lines!(ax, zonal_average,              latitude; color = (:blue, op2), label = L"\text{Ground Truth}", linewidth=2)

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
                                  xlims!(ax, extrema(hrange)...)
        else
            ax = Axis(fig[2, i];  ylabel = "", 
                                  xlabel = L"\text{Meridional heat flux [MWm}^{-2}\text{]}", 
                                  yticks = ([20, 40, 60], ["", "", ""]),
                                  xticks = xticksH[i])
                                  xlims!(ax, extrema(hrange)...)
        end



        lines!(ax, mean_zonal_average_samples ./ 1e6, latitude; color = (:red, op), label = L"\text{Generative AI}")
        for qu in [0.6, 0.7, 0.8, 0.9, 1.0]
            δlower = [quantile((zonal_average_samples[i, :]) ./ 1e6, 1-qu) for i in 1:Nlat]
            δupper = [quantile((zonal_average_samples[i, :]) ./ 1e6, qu) for i in 1:Nlat]
            band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, op))
        end
        lines!(ax, zonal_average ./ 1e6,              latitude; color = (:blue, op2), linewidth=2, label = L"\text{Ground Truth}")
        if i == 1
            axislegend(ax, position = :lt)
        end
    end

    save("Figures/zonal_average_heat_flux_fixed_depth_coarse_graining_$(XLast)_coarse_graining.png", fig)
    save("Figures/zonal_average_heat_flux_fixed_depth_coarse_graining_$(XLast).eps", fig)
end
