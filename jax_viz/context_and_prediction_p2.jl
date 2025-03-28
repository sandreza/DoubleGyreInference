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
v = zeros(128, 128)
T  = zeros(128, 128)
u = zeros(128, 128)
w = zeros(128, 128)
T_samples = zeros(128, 128, nsamples, cg_levels)
v_samples = zeros(128, 128, nsamples, cg_levels)
w_samples = zeros(128, 128, nsamples, cg_levels)
u_samples = zeros(128, 128, nsamples, cg_levels)
eta = zeros(128, 128, cg_levels)


future_year = 50
file_string = "attention_velocity_uc_production_jax_samples_"

σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevels = z_centers.(1:Nz)
level = 10

for cg in 0:(cg_levels-1)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :u, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu) 
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu) 

    (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string, cg)
    v[:, :] .= (ground_truth .* sigma .+ mu) 
    v_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu) 

    (; ground_truth, samples, mu, sigma) = jax_field(level, :w, future_year; file_string, cg)
    w[:, :] .= (ground_truth .* sigma .+ mu)
    w_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)

    (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string, cg)
    T[:, :] .= (ground_truth .* sigma .+ mu) /(α * g)
    T_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu) /(α * g) 

    (; context, mu, sigma) = jax_context(future_year; file_string, cg)
    eta[:, :, cg + 1] .= (context .* sigma .+ mu)  # free surface height
end

fields = [u, v, w, T, eta[:, :, 1]]
samples = [u_samples, v_samples, w_samples, T_samples, eta]
crange = [quantile(field[:], 0.99) for field in fields]
cmaps = [velocity_color, velocity_color, velocity_color, temperature_color, free_surface_color]
label_strings = ["u (m/s)", "v (m/s)", "w (m/s)", "T (°C)", "SSH (m)"]
lats = range(15, 75, length = 128)  # latitude range for the heatmap
lons = range(0, 60, length = 128)


title_list = ["SSH", "OcS", "AI Ensemble Mean", "AI Sample", "AI Sample", "Mean Discrepancy", "AI Std"]

factor = 200
cg = 0 
ω = 1
fig = Figure(resolution = (7 * factor, 3 * factor))
for ii in 1:4
    if ii == 1 
        titles = title_list
    else
        titles = ["" for title in eachindex(title_list)]
    end
    ax = Axis(fig[ii, 1]; title = titles[1])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, fields[5], colorrange = (-crange[5], crange[5]), colormap = cmaps[5])
    Colorbar(fig[ii, 2], hm, label = label_strings[5])
    ii < 4 ? cvals = (-crange[ii], crange[ii]) :  cvals = (0, crange[ii])
    shift = 2
    ax = Axis(fig[ii, 1 + shift]; title = titles[2])
    hidedecorations!(ax)
    heatmap!(ax, lons, lats, fields[ii], colorrange = cvals, colormap = cmaps[ii])
    ax = Axis(fig[ii, 2 + shift]; title = titles[3])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, mean(samples[ii][:, :, :, cg + 1], dims = 3)[:, :, 1], colorrange = cvals, colormap = cmaps[ii])
    ax = Axis(fig[ii, 3 + shift]; title = titles[4])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, samples[ii][:, :, ω, cg + 1], colorrange = cvals, colormap = cmaps[ii])
    ax = Axis(fig[ii, 4 + shift]; title = titles[5])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, samples[ii][:, :, end, cg + 1], colorrange = cvals, colormap = cmaps[ii])
    shift = 4
    Colorbar(fig[ii, 3 + shift],  hm, label = label_strings[ii])
    ax = Axis(fig[ii, 4 + shift]; title = titles[6])
    hidedecorations!(ax)
    hm = heatmap!(ax,  lons, lats, abs.(fields[ii] - mean(samples[ii][:, :, :, cg + 1], dims = 3)[:, :, 1]), colorrange = (0, crange[ii]/4), colormap = :viridis)
    ax = Axis(fig[ii, 5 + shift]; title = titles[7])
    hidedecorations!(ax)
    std_field = std(samples[ii][:, :, :, cg + 1], dims = 3)[:, :, 1]
    hm = heatmap!(ax, lons, lats, std_field, colorrange = (0, crange[ii]/4), colormap = :viridis)
    Colorbar(fig[ii, 6 + shift], hm, label = label_strings[ii])
end
save("Figures/test.png", fig)  # Save the figure to a file


factor = 200
cg_list = [2, 4, 5, 6]
ω = 1
fig = Figure(resolution = (7 * factor, 3 * factor))
iii = 1
for ii in 1:4
    if ii == 1 
        titles = title_list
    else
        titles = ["" for title in eachindex(title_list)]
    end
    cg = cg_list[ii]
    ax = Axis(fig[ii, 1]; title = titles[1])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, samples[5][:, :, cg + 1], colorrange = (-crange[5], crange[5]), colormap = cmaps[5])
    Colorbar(fig[ii, 2], hm, label = label_strings[5])
    cvals = (-crange[iii], crange[iii])
    shift = 2
    ax = Axis(fig[ii, 1 + shift]; title = titles[3])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, mean(samples[iii][:, :, :, cg + 1], dims = 3)[:, :, 1], colorrange = cvals, colormap = cmaps[iii])
    ax = Axis(fig[ii, 2 + shift]; title = titles[4])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, samples[iii][:, :, ω+1, cg + 1], colorrange = cvals, colormap = cmaps[iii])
    ax = Axis(fig[ii, 3 + shift]; title = titles[4])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, samples[iii][:, :, ω, cg + 1], colorrange = cvals, colormap = cmaps[iii])
    ax = Axis(fig[ii, 4 + shift]; title = titles[5])
    hidedecorations!(ax)
    hm = heatmap!(ax, lons, lats, samples[iii][:, :, end, cg + 1], colorrange = cvals, colormap = cmaps[iii])
    shift = 4
    Colorbar(fig[ii, 3 + shift],  hm, label = label_strings[iii])
    #=
    ax = Axis(fig[ii, 4 + shift]; title = titles[6])
    hidedecorations!(ax)
    hm = heatmap!(ax,  lons, lats, abs.(fields[iii] - mean(samples[iii][:, :, :, cg + 1], dims = 3)[:, :, 1]), colorrange = (0, crange[iii]/4), colormap = :viridis)
    =#
    ax = Axis(fig[ii, 4 + shift]; title = titles[7])
    hidedecorations!(ax)
    std_field = std(samples[iii][:, :, :, cg + 1], dims = 3)[:, :, 1]
    hm = heatmap!(ax, lons, lats, std_field, colorrange = (0, crange[iii]/4), colormap = :viridis)
    Colorbar(fig[ii, 5 + shift], hm, label = label_strings[iii])
end
save("Figures/test_2.png", fig)  # Save the figure to a file

##
mean_fields = [mean(samples[iii][:, :, :, 8], dims = 3)[:, :, 1] for iii in 1:4]
std_fields = [std(samples[iii][:, :, :, 8], dims = 3)[:, :, 1] for iii in 1:4]
data_tuple = return_data_file(3) # level 3 from before is level 10 now

mus = reshape([data_tuple.u_mean, data_tuple.v_mean, data_tuple.w_mean, data_tuple.b_mea/ ( α * g)], (1, 1, 4))
sigmas = reshape([data_tuple.u_2std, data_tuple.v_2std, data_tuple.w_2std, data_tuple.b_2std/( α * g)], (1, 1, 4))
ocs_mean = (data_tuple.mean_field[:, :, 1:4] .* sigmas) .+ mus
ocs_std = (data_tuple.std_field[:, :, 1:4] .* sigmas)
fig = Figure()
for i in 1:4 
    i < 4 ? cvals = (-crange[i], crange[i]) :  cvals = (0, crange[i])
    i < 4 ? sfactor = 1 : sfactor = 4
    ax = Axis(fig[i, 1])
    hidedecorations!(ax)
    heatmap!(mean_fields[i], colorrange = cvals, colormap = cmaps[i])
    ax = Axis(fig[i, 2])
    hidedecorations!(ax)
    heatmap!(ocs_mean[:, :, i], colorrange = cvals, colormap = cmaps[i])
    ax = Axis(fig[i, 3])
    hidedecorations!(ax)
    hm = heatmap!(ax, std_fields[i], colorrange = (0, crange[i]/2 / sfactor), colormap = :viridis)
    ax = Axis(fig[i, 4])
    hidedecorations!(ax)
    hm = heatmap!(ax, ocs_std[:, :, i], colorrange = (0, crange[i]/2 / sfactor), colormap = :viridis)
end
save("Figures/test_3.png", fig)