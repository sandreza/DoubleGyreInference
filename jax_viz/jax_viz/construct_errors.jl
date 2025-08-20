using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5, Random
using LaTeXStrings
α = 2e-4
g = 9.81

level_indices = [14, 12, 10, 3]
total_levels = length(level_indices)
Nz = 15
Lz = 1800
nsamples = 100

cg_levels = 8
sorted_vlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_data  = zeros(128, 128, total_levels)
sorted_ulevels_data = zeros(128, 128, total_levels)
sorted_wlevels_data = zeros(128, 128, total_levels)
sorted_Tlevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)
sorted_vlevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)
sorted_wlevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)
sorted_ulevels_samples = zeros(128, 128, total_levels, nsamples, cg_levels)

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

for cg in 0:(cg_levels-1)
    for level in ProgressBar(1:total_levels)
        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :u, future_year; file_string, cg)
        sorted_ulevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
        sorted_ulevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu) 

        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :v, future_year; file_string, cg)
        sorted_vlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
        sorted_vlevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu) 

        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :w, future_year; file_string, cg)
        sorted_wlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu)
        sorted_wlevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu)

        (; ground_truth, samples, mu, sigma) = jax_field(level_indices[level], :b, future_year; file_string, cg)
        sorted_Tlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) /(α * g)
        sorted_Tlevels_samples[:, :, level, :, cg + 1] .= (samples .* sigma .+ mu) /(α * g)
    end
end


ground_truth_error_u = zeros(total_levels, cg_levels, 100)
ground_truth_error_v = zeros(total_levels, cg_levels, 100)
ground_truth_error_w = zeros(total_levels, cg_levels, 100)
ground_truth_error_T = zeros(total_levels, cg_levels, 100)

shuffle_error_u = zeros(total_levels, cg_levels, 100)
shuffle_error_v = zeros(total_levels, cg_levels, 100)
shuffle_error_w = zeros(total_levels, cg_levels, 100)
shuffle_error_T = zeros(total_levels, cg_levels, 100)
scales = [1e-2, 1e-2, 1e-6, 1]
perm = circshift(1:100, 1)
for cg in 0:(cg_levels-1)
    for level in 1:total_levels
        ground_truth_error_u[level, cg + 1, :] = mean(abs.( sorted_ulevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_ulevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[1]
        ground_truth_error_v[level, cg + 1, :] = mean(abs.( sorted_vlevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_vlevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[2]
        ground_truth_error_w[level, cg + 1, :] = mean(abs.( sorted_wlevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_wlevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[3]
        ground_truth_error_T[level, cg + 1, :] = mean(abs.( sorted_Tlevels_samples[:, :, level, :, cg + 1] .- reshape(sorted_Tlevels_data[:, :, level], (128, 128, 1))), dims = (1, 2) )[:] / scales[4]

        shuffle_error_u[level, cg + 1, :] = mean(abs.(sorted_ulevels_samples[:, :, level, perm, cg + 1] - sorted_ulevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[1]
        shuffle_error_v[level, cg + 1, :] = mean(abs.(sorted_vlevels_samples[:, :, level, perm, cg + 1] - sorted_vlevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[2]
        shuffle_error_w[level, cg + 1, :] = mean(abs.(sorted_wlevels_samples[:, :, level, perm, cg + 1] - sorted_wlevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[3]
        shuffle_error_T[level, cg + 1, :] = mean(abs.(sorted_Tlevels_samples[:, :, level, perm, cg + 1] - sorted_Tlevels_samples[:, :, level, :, cg + 1]), dims = (1, 2))[:] / scales[4]
    end
end

##
colors = [:blue, :orange, :green, :brown]
vals = zlevels[level_indices]
# hard code it this is a potential source of error
label_names = [L"79 \text{ m}",
L"213 \text{ m}",
L"387 \text{ m}",
L"1355 \text{ m}"]

ticks(t) = (t, [L"%$i" for i in t])

cgs = collect(1:cg_levels)
xticks = (cgs, [L"2^0", L"2^1", L"2^2", L"2^3", L"2^4", L"2^5", L"2^6", L"2^7"])

ylims = [(0, 3), (0, 4), (0, 11), (0, 1.5) ]
ylimfactor = [2, 2, 1, 4]
yticks_p = [range(ylims[i]..., length = ceil(Int, ylimfactor[i]*ylims[i][end] + 1))[2:end-1] for i in eachindex(ylims)]
yticks = (ticks(yticks_p[1]), 
          ticks(yticks_p[2]),
          ticks(yticks_p[3]),
          ticks(yticks_p[4]))

qus = [0.6, 0.7, 0.8, 0.9]
op = 0.5
op2 = 0.1

normlabel = ["L1", "L2", "Linfty" ]
fieldnames = ["U", "V", "W", "T"]
# units = [L"\text{cm/s}", L"\text{cm/s}", L"\mu\text{m/s}", L"^\circ\text{C}"]
units = [L"\text{[cm/s]}", L"\text{[cm/s]}", L"\text{[}\mu\text{m/s]}", L"\text{[}^\circ\text{C]}", L"\text{[m]}"]
xlabel = L"\text{Coarse-Graining Factor}"

title_strings_1 = [L"\text{\textbf{OsC AI Discrepancy U}}",
                   L"\text{\textbf{OsC AI Discrepancy V}}",
                   L"\text{\textbf{OsC AI Discrepancy W}}",
                   L"\text{\textbf{OsC AI Discrepancy T}}"]
title_strings_2 = [L"\text{\textbf{AI Shuffle Discrepancy U}}",
                   L"\text{\textbf{AI Shuffle Discrepancy V}}",
                   L"\text{\textbf{AI Shuffle Discrepancy W}}",
                   L"\text{\textbf{AI Shuffle Discrepancy T}}"]

fields1 = [ground_truth_error_u, ground_truth_error_v, ground_truth_error_w, ground_truth_error_T]
fields2 = [shuffle_error_u, shuffle_error_v, shuffle_error_w, shuffle_error_T]
##
factor = 500
fig = Figure(resolution = (2 * factor, factor))
for ii in 1:4
ax = Axis(fig[1, ii]; title = title_strings_1[ii], xticks=xticks, yticks=yticks[ii], ylabel = units[ii], xlabel  = xlabel)
state = fields1[ii]
for level in 1:total_levels
    field = state[level, :, :]
    for qu in qus
        qu_lower = [quantile(field[j, :], 1-qu) for j in 1:cg_levels]
        qu_uppper = [quantile(field[j, :], qu) for j in 1:cg_levels]
        band!(ax, Point.(cgs, qu_lower), Point.(cgs, qu_uppper); color = (colors[level], op2))
    end
    meanvals = mean(field, dims = 2)[:]
    scatterlines!(ax, cgs, meanvals, color = (colors[level], op), label = label_names[level], marker = :xcross)
end
if ii == 4
    axislegend(ax, position = :lt)
end
ylims!(ax, ylims[ii]...)
ax = Axis(fig[2, ii]; title = title_strings_2[ii], xticks=xticks, yticks=yticks[ii],  ylabel = units[ii], xlabel  = xlabel)
state = fields2[ii]
for level in 1:total_levels
    field = state[level, :, :]
    for qu in qus
        qu_lower = [quantile(field[j, :], 1-qu) for j in 1:cg_levels]
        qu_uppper = [quantile(field[j, :], qu) for j in 1:cg_levels]
        band!(ax, Point.(cgs, qu_lower), Point.(cgs, qu_uppper); color = (colors[level], op2))
    end
    meanvals = mean(field, dims = 2)[:]
    scatterlines!(ax, cgs, meanvals, color = (colors[level], op), label = label_names[level], marker = :xcross)
end
ylims!(ax, ylims[ii]...)
end

save("Figures/osc_ai_discrepancy.png", fig)