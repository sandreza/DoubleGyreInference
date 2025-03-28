using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5, Random
α = 2e-4
g = 9.81

total_levels = Nz = 15
Lz = 1800
nsamples = 100
sorted_vlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_data  = zeros(128, 128, total_levels)
sorted_ulevels_data = zeros(128, 128, total_levels)
sorted_wlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_vlevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_wlevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_ulevels_samples = zeros(128, 128, total_levels, nsamples)

future_year = 50
cg = 0 # coarse-graining level
file_string = "attention_velocity_uc_production_jax_samples_"

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
hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)

for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :u, future_year; file_string, cg)
    sorted_ulevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_ulevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 

    (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string, cg)
    sorted_vlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_vlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 

    (; ground_truth, samples, mu, sigma) = jax_field(level, :w, future_year; file_string, cg)
    sorted_wlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu)
    sorted_wlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu)

    (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string, cg)
    sorted_Tlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) /(α * g)
    sorted_Tlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) /(α * g)
end


function field_correlation_data(field1, field2; flatten = true)
    if flatten
        return [sum(field1[:, :, j] .* field2[:, :, k]) ./ sqrt.(sum(field1[:, :, j] .^2) .* sum(field2[:, :, k] .^2)) for j in 1:Nz, k in 1:Nz]
    else
        return [(sum(field1[:, :, j, :] .* field2[:, :, k, :], dims = (1, 2)) ./ sqrt.(sum(field1[:, :, j, :] .^2, dims = (1, 2)) .* sum(field2[:, :, k, :] .^2, dims = (1, 2))))[:] for j in 1:Nz, k in 1:Nz]
    end
end
field_correlation_data(field; flatten = true) = field_correlation_data(field, field; flatten = flatten)

fields_data = [sorted_ulevels_data, sorted_vlevels_data, sorted_wlevels_data, sorted_Tlevels_data];
fields_samples = [sorted_ulevels_samples, sorted_vlevels_samples, sorted_wlevels_samples, sorted_Tlevels_samples];
data_correlations = []
sample_correlations = []
sample_data_correlations = []
sample_shuffle_correlations = []
perm = circshift(1:100, 1)
for i in ProgressBar(1:4)
    push!(data_correlations, field_correlation_data(fields_data[i]))
    push!(sample_correlations, field_correlation_data(fields_samples[i], flatten = false))
    push!(sample_data_correlations, field_correlation_data(fields_data[i], fields_samples[i], flatten = false))
    push!(sample_shuffle_correlations, field_correlation_data(fields_samples[i][:, :, :, :], fields_samples[i][:, :, :, perm], flatten = false))
end
##
colors = [:blue, :orange, :green, :brown]
label_names = ["u", "v", "w", "T"]
qus = [0.6, 0.7, 0.8, 0.9]
op = 0.5
op2 = 0.1
xbottom = 0.3
xtop = 1.05

factor = 340
fig = Figure(resolution = (4*factor, 1*factor))
ax = Axis(fig[1, 1]; title = "OcS Correlation with OcS Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
# data
for i in 1:4
    scatterlines!(ax, data_correlations[i][Nz, :], zlevels, color = (colors[i], op), label = label_names[i])
end
axislegend(ax, position = :lt, orientation = :horizontal)
xlims!(ax, (xbottom, xtop))
# samples 
ax = Axis(fig[1, 2]; title = "AI Correlation with AI Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = sample_correlations[i][Nz, :]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :star6)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 3]; title = "AI Correlation with OcS", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = [sample_data_correlations[i][j, j] for j in 1:Nz]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :hexagon)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 4]; title = "AI Shuffle Correlation", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = [sample_shuffle_correlations[i][j, j] for j in 1:Nz]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :xcross)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)
save("Figures/jax_depth_correlation_$(future_year)_$(cg)_and_ai.png", fig)

##
cg = 7
for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :u, future_year; file_string, cg)
    sorted_ulevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_ulevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 

    (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string, cg)
    sorted_vlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_vlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 

    (; ground_truth, samples, mu, sigma) = jax_field(level, :w, future_year; file_string, cg)
    sorted_wlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu)
    sorted_wlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu)

    (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string, cg)
    sorted_Tlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) /(α * g)
    sorted_Tlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) /(α * g)
end

fields_data = [sorted_ulevels_data, sorted_vlevels_data, sorted_wlevels_data, sorted_Tlevels_data];
fields_samples = [sorted_ulevels_samples, sorted_vlevels_samples, sorted_wlevels_samples, sorted_Tlevels_samples];

for i in ProgressBar(1:4)
    push!(data_correlations, field_correlation_data(fields_data[i]))
    push!(sample_correlations, field_correlation_data(fields_samples[i], flatten = false))
    push!(sample_data_correlations, field_correlation_data(fields_data[i], fields_samples[i], flatten = false))
    push!(sample_shuffle_correlations, field_correlation_data(fields_samples[i][:, :, :, :], fields_samples[i][:, :, :, perm], flatten = false))
end
##

colors = [:blue, :orange, :green, :brown]
label_names = ["u", "v", "w", "T"]
qus = [0.6, 0.7, 0.8, 0.9]
op = 0.5
op2 = 0.1

xbottom = 0.0
xtop = 1.10

factor = 340
fig = Figure(resolution = (6*factor, 1*factor))
ax = Axis(fig[1, 1]; title = "OcS Correlation with OcS Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
# data
for i in 1:4
    scatterlines!(ax, data_correlations[i][Nz, :], zlevels, color = (colors[i], op), label = label_names[i])
end
axislegend(ax, position = :lt, orientation = :horizontal)
xlims!(ax, (xbottom, xtop))
# samples 
ax = Axis(fig[1, 2]; title = "AI Correlation with AI Surface Field", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = sample_correlations[i][Nz, :]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :star6)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 3]; title = "AI Correlation with OcS: Full SSH", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = [sample_data_correlations[i][j, j] for j in 1:Nz]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :hexagon)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 4]; title = "AI Shuffle Correlation: Full SSH", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = [sample_shuffle_correlations[i][j, j] for j in 1:Nz]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :xcross)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 5]; title = "AI Correlation with OcS: SSH Mean", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = [sample_data_correlations[i+4][j, j] for j in 1:Nz]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :hexagon)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 6]; title = "AI Shuffle Correlation: SSH Mean", xlabel = "Correlation", ylabel = "Depth [m]")
for i in 1:4
    val = [sample_shuffle_correlations[i+4][j, j] for j in 1:Nz]
    meanvals = [mean(val[j][:]) for j in 1:Nz]
    scatterlines!(ax, meanvals, zlevels, color = (colors[i], op), label = label_names[i], marker = :xcross)
    for qu in qus
        qu_lower = [quantile(val[j][:], 1-qu) for j in 1:Nz]
        qu_uppper = [quantile(val[j][:], qu) for j in 1:Nz]
        band!(ax, Point.(qu_lower, zlevels), Point.(qu_uppper, zlevels); color = (colors[i], op2))
    end
end
xlims!(ax, (xbottom, xtop))
hideydecorations!(ax; hiding_options...)


save("Figures/jax_depth_correlation_$(future_year)_ai_more.png", fig)