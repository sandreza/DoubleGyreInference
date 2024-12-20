using CairoMakie, Statistics, Random, LinearAlgebra
Random.seed!(1234)

function symmetric_quantile_range(data, q)
    quantile_value = quantile(abs.(data)[:], q)
    return (-quantile_value, quantile_value)
end

function quantile_range(data, q)
    quantile_value_min = quantile(data[:], q)
    quantile_value_max = quantile(data[:], 1 - q)
    return sort([quantile_value_min, quantile_value_max])
end

function clipped_quantile_range(data, q)
    return (0, quantile(data[:], q))
end


lons = Float32.(range(0, 60, length = 128))
lats = Float32.(range(15, 74, length = 128))

qu = 0.95
# color and quantile functions 
quantile_functions = [symmetric_quantile_range, symmetric_quantile_range, symmetric_quantile_range, quantile_range]
colors = [:balance, :balance, :balance, :thermometer]

fig = Figure(resolution = (2500, 2000)) 
for field_index in 1:4
    ax = Axis(fig[field_index, 1]; title = "Ground Truth")
    field = tupled_data.field_2[:, :, field_index]
    qf = quantile_functions[field_index]
    field_range =  qf(field, qu)
    heatmap!(ax, lons, lats, field; colormap = colors[field_index], colorrange = field_range, interpolate = false)

    ax = Axis(fig[field_index, 2]; title = "Sample Average")
    field = tupled_data.averaged_samples_2[:, :, field_index]
    heatmap!(ax, lons, lats, field; colormap = colors[field_index], colorrange = field_range, interpolate = false)

    ax = Axis(fig[field_index, 3]; title = "Sample Average - Ground Truth")
    field = tupled_data.field_2[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]
    field_range = symmetric_quantile_range(tupled_data.field_2, qu)
    heatmap!(ax, lons, lats, field; colormap = :balance, colorrange = field_range, interpolate = false)

    ax = Axis(fig[field_index, 4]; title = "Sample Standard Deviation")
    field = tupled_data.std_samples_2[:, :, field_index]
    field_range_std =  clipped_quantile_range(field, qu)
    heatmap!(ax, lons, lats, field; colormap = :viridis, colorrange = field_range_std, interpolate = false)

    ax = Axis(fig[field_index, 5]; title = "Sample 1")
    field = tupled_data.samples_2[:, :, field_index, 1]
    heatmap!(ax, lons, lats, field; colormap = colors[field_index], colorrange = field_range, interpolate = false)
    ax = Axis(fig[field_index, 6]; title = "Sample 2")
    field = tupled_data.samples_2[:, :, field_index, 2]
    heatmap!(ax, lons, lats, field; colormap = colors[field_index], colorrange = field_range, interpolate = false)
    

end
save("Figures/fields_level_index_$(level_index)_factor_$(factor).png", fig)