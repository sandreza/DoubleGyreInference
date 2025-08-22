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
scales_factor = [1e-2, 1e-2, 1e-6, (α * g) ]
for cg in 0:7
    field_symbol = :u
    (; ground_truth, samples, mu, sigma) = jax_field(level, field_symbol, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu)  / scales_factor[1]
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)  / scales_factor[1]

    directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/" 
    new_file_string = "_traditional_cg_$(cg).hdf5"
    state_index = DoubleGyreInference.jax_symbol_to_index(field_symbol)
    field_index = 4 * (level - 1) + state_index + 1
    hfile = h5open(directory * new_file_string, "r")
    mean_sample = read(hfile["output"])[:, :, field_index]
    ground_truth_2 = read(hfile["ground_truth"])[:, :, field_index]
    u_mean = (mean_sample .* sigma .+ mu ) / scales_factor[1]
    close(hfile)

    generative_mean = mean(u_samples[:, :, :, cg + 1], dims = 3)[:, :, 1]

    (; context, mu, sigma) = jax_context(future_year; file_string, cg)
    eta = (context .* sigma .+ mu)[:, :, 1]
    lats = range(15, 75, length = 128)  # latitude range for the heatmap
    lons = range(0, 60, length = 128)

    crange = quantile(u[:], 0.99)

    fig = Figure(resolution = (7 * 250, 250))
    ax = Axis(fig[1, 1]; title = "Context")
    hm = heatmap!(ax, lons, lats, eta, colorrange = (-1,1), colormap = free_surface_color)
    hidedecorations!(ax)
    Colorbar(fig[1, 2], hm, label = "η (m)")
    shift = 2
    ax = Axis(fig[1, 1 + shift]; title = "Truth")
    heatmap!(ax, lons, lats, u, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    ax = Axis(fig[1, 2 + shift]; title = "Generative Mean")
    heatmap!(ax, lons, lats, generative_mean , colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    ax = Axis(fig[1, 3 + shift]; title = "Traditional Model")
    hm = heatmap!(ax, lons, lats, u_mean, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    Colorbar(fig[1, 4 + shift], hm, label = "u (m/s)")
    ax = Axis(fig[1, 5 + shift]; title = "Generative Mean - Truth")
    heatmap!(ax, lons, lats, abs.(generative_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    ax = Axis(fig[1, 6 + shift]; title = "Traditional Model - Truth")
    hm = heatmap!(ax, lons, lats, abs.(u_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    ax = Axis(fig[1, 7 + shift]; title = "Generative Mean - Traditional")
    heatmap!(ax, lons, lats, abs.(generative_mean - u_mean), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    Colorbar(fig[1, 8 + shift], hm, label = "u (m/s)")
    println("The mean absolute error of the generative mean is ", mean(abs.(generative_mean - u)))
    println("The mean absolute error of the traditional mean is ", mean(abs.(u_mean - u)))
    save("Figures/deterministic_$cg.png", fig)
end



fig = Figure(resolution = (7 * 250, 8*250))
for cg in 0:7
    field_symbol = :u
    (; ground_truth, samples, mu, sigma) = jax_field(level, field_symbol, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu)  / scales_factor[1]
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)  / scales_factor[1]

    directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/" 
    new_file_string = "_traditional_cg_$(cg).hdf5"
    state_index = DoubleGyreInference.jax_symbol_to_index(field_symbol)
    field_index = 4 * (level - 1) + state_index + 1
    hfile = h5open(directory * new_file_string, "r")
    mean_sample = read(hfile["output"])[:, :, field_index]
    ground_truth_2 = read(hfile["ground_truth"])[:, :, field_index]
    u_mean = (mean_sample .* sigma .+ mu ) / scales_factor[1]
    close(hfile)

    generative_mean = mean(u_samples[:, :, :, cg + 1], dims = 3)[:, :, 1]

    (; context, mu, sigma) = jax_context(future_year; file_string, cg)
    eta = (context .* sigma .+ mu)[:, :, 1]
    lats = range(15, 75, length = 128)  # latitude range for the heatmap
    lons = range(0, 60, length = 128)

    crange = quantile(u[:], 0.99)


    if cg == 0
        ax = Axis(fig[1 + cg, 1]; title = "Context")
    else 
        ax = Axis(fig[1 + cg, 1])
    end
    hm = heatmap!(ax, lons, lats, eta, colorrange = (-1,1), colormap = free_surface_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 2], hm, label = "η (m)")
    shift = 2
    if cg == 0
        ax = Axis(fig[1 + cg, 1 + shift]; title = "Truth")
    else 
        ax = Axis(fig[1 + cg, 1 + shift])
    end
    heatmap!(ax, lons, lats, u, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 2 + shift]; title = "Generative Mean")
    else 
        ax = Axis(fig[1 + cg, 2 + shift])
    end
    heatmap!(ax, lons, lats, generative_mean , colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 3 + shift]; title = "Traditional Model")
    else 
        ax = Axis(fig[1 + cg, 3 + shift])
    end
    hm = heatmap!(ax, lons, lats, u_mean, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 4 + shift], hm, label = "u (m/s)")
    if cg == 0
        ax = Axis(fig[1 + cg, 5 + shift]; title = "Generative Mean - Truth")
    else 
        ax = Axis(fig[1 + cg, 5 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 6 + shift]; title = "Traditional Model - Truth")
    else 
        ax = Axis(fig[1 + cg, 6 + shift])
    end
    hm = heatmap!(ax, lons, lats, abs.(u_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 7 + shift]; title = "Generative Mean - Traditional")
    else 
        ax = Axis(fig[1 + cg, 7 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u_mean), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "u (m/s)")
    else 
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "u (m/s)")
    end
    println("The mean absolute error of the generative mean is ", norm(generative_mean - u))
    println("The mean absolute error of the traditional mean is ", norm(u_mean - u))
end
save("Figures/deterministic_vs_generative_u.png", fig)


fig = Figure(resolution = (7 * 250, 8*250))
for cg in 0:7
    field_symbol = :w
    (; ground_truth, samples, mu, sigma) = jax_field(level, field_symbol, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu)  / scales_factor[1]
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)  / scales_factor[1]

    directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/" 
    new_file_string = "_traditional_cg_$(cg).hdf5"
    state_index = DoubleGyreInference.jax_symbol_to_index(field_symbol)
    field_index = 4 * (level - 1) + state_index + 1
    hfile = h5open(directory * new_file_string, "r")
    mean_sample = read(hfile["output"])[:, :, field_index]
    ground_truth_2 = read(hfile["ground_truth"])[:, :, field_index]
    u_mean = (mean_sample .* sigma .+ mu ) / scales_factor[1]
    close(hfile)

    generative_mean = mean(u_samples[:, :, :, cg + 1], dims = 3)[:, :, 1]

    (; context, mu, sigma) = jax_context(future_year; file_string, cg)
    eta = (context .* sigma .+ mu)[:, :, 1]
    lats = range(15, 75, length = 128)  # latitude range for the heatmap
    lons = range(0, 60, length = 128)

    crange = quantile(u[:], 0.99)


    if cg == 0
        ax = Axis(fig[1 + cg, 1]; title = "Context")
    else 
        ax = Axis(fig[1 + cg, 1])
    end
    hm = heatmap!(ax, lons, lats, eta, colorrange = (-1,1), colormap = free_surface_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 2], hm, label = "η (m)")
    shift = 2
    if cg == 0
        ax = Axis(fig[1 + cg, 1 + shift]; title = "Truth")
    else 
        ax = Axis(fig[1 + cg, 1 + shift])
    end
    heatmap!(ax, lons, lats, u, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 2 + shift]; title = "Generative Mean")
    else 
        ax = Axis(fig[1 + cg, 2 + shift])
    end
    heatmap!(ax, lons, lats, generative_mean , colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 3 + shift]; title = "Traditional Model")
    else 
        ax = Axis(fig[1 + cg, 3 + shift])
    end
    hm = heatmap!(ax, lons, lats, u_mean, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 4 + shift], hm, label = "w (m/s)")
    if cg == 0
        ax = Axis(fig[1 + cg, 5 + shift]; title = "Generative Mean - Truth")
    else 
        ax = Axis(fig[1 + cg, 5 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 6 + shift]; title = "Traditional Model - Truth")
    else 
        ax = Axis(fig[1 + cg, 6 + shift])
    end
    hm = heatmap!(ax, lons, lats, abs.(u_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 7 + shift]; title = "Generative Mean - Traditional")
    else 
        ax = Axis(fig[1 + cg, 7 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u_mean), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "w (m/s)")
    else 
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "w (m/s)")
    end
    println("The mean absolute error of the generative mean is ", norm(generative_mean - u))
    println("The mean absolute error of the traditional mean is ", norm(u_mean - u))
end
save("Figures/deterministic_vs_generative_w.png", fig)



fig = Figure(resolution = (7 * 250, 8*250))
for cg in 0:7
    field_symbol = :v
    (; ground_truth, samples, mu, sigma) = jax_field(level, field_symbol, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu)  / scales_factor[1]
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)  / scales_factor[1]

    directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/" 
    new_file_string = "_traditional_cg_$(cg).hdf5"
    state_index = DoubleGyreInference.jax_symbol_to_index(field_symbol)
    field_index = 4 * (level - 1) + state_index + 1
    hfile = h5open(directory * new_file_string, "r")
    mean_sample = read(hfile["output"])[:, :, field_index]
    ground_truth_2 = read(hfile["ground_truth"])[:, :, field_index]
    u_mean = (mean_sample .* sigma .+ mu ) / scales_factor[1]
    close(hfile)

    generative_mean = mean(u_samples[:, :, :, cg + 1], dims = 3)[:, :, 1]

    (; context, mu, sigma) = jax_context(future_year; file_string, cg)
    eta = (context .* sigma .+ mu)[:, :, 1]
    lats = range(15, 75, length = 128)  # latitude range for the heatmap
    lons = range(0, 60, length = 128)

    crange = quantile(u[:], 0.99)


    if cg == 0
        ax = Axis(fig[1 + cg, 1]; title = "Context")
    else 
        ax = Axis(fig[1 + cg, 1])
    end
    hm = heatmap!(ax, lons, lats, eta, colorrange = (-1,1), colormap = free_surface_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 2], hm, label = "η (m)")
    shift = 2
    if cg == 0
        ax = Axis(fig[1 + cg, 1 + shift]; title = "Truth")
    else 
        ax = Axis(fig[1 + cg, 1 + shift])
    end
    heatmap!(ax, lons, lats, u, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 2 + shift]; title = "Generative Mean")
    else 
        ax = Axis(fig[1 + cg, 2 + shift])
    end
    heatmap!(ax, lons, lats, generative_mean , colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 3 + shift]; title = "Traditional Model")
    else 
        ax = Axis(fig[1 + cg, 3 + shift])
    end
    hm = heatmap!(ax, lons, lats, u_mean, colorrange = (-crange, crange), colormap = velocity_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 4 + shift], hm, label = "v (m/s)")
    if cg == 0
        ax = Axis(fig[1 + cg, 5 + shift]; title = "Generative Mean - Truth")
    else 
        ax = Axis(fig[1 + cg, 5 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 6 + shift]; title = "Traditional Model - Truth")
    else 
        ax = Axis(fig[1 + cg, 6 + shift])
    end
    hm = heatmap!(ax, lons, lats, abs.(u_mean - u), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 7 + shift]; title = "Generative Mean - Traditional")
    else 
        ax = Axis(fig[1 + cg, 7 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u_mean), colorrange = (0, crange/2), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "v (m/s)")
    else 
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "v (m/s)")
    end
    println("The mean absolute error of the generative mean is ", norm(generative_mean - u))
    println("The mean absolute error of the traditional mean is ", norm(u_mean - u))
end
save("Figures/deterministic_vs_generative_v.png", fig)



fig = Figure(resolution = (7 * 250, 8*250))
for cg in 0:7
    field_symbol = :b
    (; ground_truth, samples, mu, sigma) = jax_field(level, field_symbol, future_year; file_string, cg)
    u[:, :] .= (ground_truth .* sigma .+ mu)  / scales_factor[1]
    u_samples[:, :, :, cg + 1] .= (samples .* sigma .+ mu)  / scales_factor[1]

    directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/" 
    new_file_string = "_traditional_cg_$(cg).hdf5"
    state_index = DoubleGyreInference.jax_symbol_to_index(field_symbol)
    field_index = 4 * (level - 1) + state_index + 1
    hfile = h5open(directory * new_file_string, "r")
    mean_sample = read(hfile["output"])[:, :, field_index]
    ground_truth_2 = read(hfile["ground_truth"])[:, :, field_index]
    u_mean = (mean_sample .* sigma .+ mu ) / scales_factor[1]
    close(hfile)

    generative_mean = mean(u_samples[:, :, :, cg + 1], dims = 3)[:, :, 1]

    (; context, mu, sigma) = jax_context(future_year; file_string, cg)
    eta = (context .* sigma .+ mu)[:, :, 1]
    lats = range(15, 75, length = 128)  # latitude range for the heatmap
    lons = range(0, 60, length = 128)

    crange_up = quantile(u[:], 0.99)
    crange_down = quantile(u[:], 0.01)


    if cg == 0
        ax = Axis(fig[1 + cg, 1]; title = "Context")
    else 
        ax = Axis(fig[1 + cg, 1])
    end
    hm = heatmap!(ax, lons, lats, eta, colorrange = (-1,1), colormap = free_surface_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 2], hm, label = "η (m)")
    shift = 2
    if cg == 0
        ax = Axis(fig[1 + cg, 1 + shift]; title = "Truth")
    else 
        ax = Axis(fig[1 + cg, 1 + shift])
    end
    heatmap!(ax, lons, lats, u, colorrange = (crange_down, crange_up), colormap = temperature_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 2 + shift]; title = "Generative Mean")
    else 
        ax = Axis(fig[1 + cg, 2 + shift])
    end
    heatmap!(ax, lons, lats, generative_mean , colorrange = (crange_down, crange_up), colormap = temperature_color)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 3 + shift]; title = "Traditional Model")
    else 
        ax = Axis(fig[1 + cg, 3 + shift])
    end
    hm = heatmap!(ax, lons, lats, u_mean, colorrange = (crange_down, crange_up), colormap = temperature_color)
    hidedecorations!(ax)
    Colorbar(fig[1 + cg, 4 + shift], hm, label = "T (°C)")
    if cg == 0
        ax = Axis(fig[1 + cg, 5 + shift]; title = "Generative Mean - Truth")
    else 
        ax = Axis(fig[1 + cg, 5 + shift])
    end
    error_val = 0.25
    heatmap!(ax, lons, lats, abs.(generative_mean - u), colorrange = (0, error_val), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 6 + shift]; title = "Traditional Model - Truth")
    else 
        ax = Axis(fig[1 + cg, 6 + shift])
    end
    hm = heatmap!(ax, lons, lats, abs.(u_mean - u), colorrange = (0, error_val), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        ax = Axis(fig[1 + cg, 7 + shift]; title = "Generative Mean - Traditional")
    else 
        ax = Axis(fig[1 + cg, 7 + shift])
    end
    heatmap!(ax, lons, lats, abs.(generative_mean - u_mean), colorrange = (0, error_val), colormap = :viridis)
    hidedecorations!(ax)
    if cg == 0
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "T (°C)")
    else 
        Colorbar(fig[1 + cg, 8 + shift], hm, label = "T (°C)")
    end
    println("The mean absolute error of the generative mean is ", norm(generative_mean - u))
    println("The mean absolute error of the traditional mean is ", norm(u_mean - u))
end
save("Figures/deterministic_vs_generative_T.png", fig)