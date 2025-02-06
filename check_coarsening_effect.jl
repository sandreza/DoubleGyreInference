using HDF5, DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie
using LaTeXStrings
r_pref = DoubleGyreInference.return_prefix
sampled_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"
oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
file_end_1 = "generative_samples.hdf5"
field_end_2 = "double_gyre_losses.hdf5"
figure_directory = "Figures/"
isdir(figure_directory) || mkdir(figure_directory)


level_errors = []
scales = []
zlevels = []

factors = [2^k for k in 0:7]
normlabel = ["L1", "L2", "Linfty" ]
fieldnames = ["U", "V", "W", "T"]
units = ["m/s", "m/s", "m/s", "Kelvin"]
xlabel = "Log2 Coarse-Graining"

for level_index in ProgressBar([1, 2, 3, 5, 7])
    tds = []
    for factor in ProgressBar(factors)
        filename = sampled_data_directory * r_pref(level_index, factor) * file_end_1

        @info "loading samples from $filename"
        hfile = h5open(filename, "r")
        averaged_samples_1 = read(hfile["averaged samples 1"])
        averaged_samples_2 = read(hfile["averaged samples 2"])
        std_samples_1 = read(hfile["std samples 1"])
        std_samples_2 = read(hfile["std samples 2"])
        context_field_1 = read(hfile["context field 1"])
        context_field_2 = read(hfile["context field 2"])
        sample_index_1 = read(hfile["sample index 1"])
        sample_index_2 = read(hfile["sample index 2"])
        N = read(hfile["last training index"])
        close(hfile)
        tupled_data = (; averaged_samples_1, averaged_samples_2, std_samples_1, std_samples_2, context_field_1, context_field_2, sample_index_1, sample_index_2, N)
        push!(tds, tupled_data)
        @info "loading field"
    end

    sample_index_1 = tds[1].sample_index_1
    sample_index_2 = tds[1].sample_index_2
    N = tds[1].N

    hfile = h5open(oceananigans_data_directory  * r_pref(level_index, 1)[1:end-3] * ".hdf5", "r")
    total_field = read(hfile["field"])
    field_1 = total_field[:, :, :, sample_index_1]
    field_2 = total_field[:, :, :, sample_index_2]
    mean_field = mean(total_field[:, :,:,  1:N], dims = 4)
    std_field = std(total_field[:, :,:,  1:N], dims = 4)
    zlevel = read(hfile["zlevel"])
    close(hfile)
    push!(zlevels, zlevel)

    error_list = zeros(length(tds), 4, 3)
    mean_distance = zeros(length(tds), 4, 3)
    std_distance = zeros(length(tds), 4, 3)
    for coarse_graining_index in eachindex(tds)
        for field_index in 1:4
            Δ = tds[coarse_graining_index].averaged_samples_2[:, :, field_index] - field_2[:, :, field_index]
            [error_list[coarse_graining_index, field_index, k] = norm(Δ, kk) / norm(ones(128,128), kk) for (k,kk) in enumerate([1, 2, Inf])]

            Δ = tds[coarse_graining_index].averaged_samples_2[:, :, field_index] - mean_field[:, :, field_index]
            [mean_distance[coarse_graining_index, field_index, k] = norm(Δ, kk) / norm(ones(128,128), kk) for (k,kk) in enumerate([1, 2, Inf])]

            Δ = tds[coarse_graining_index].std_samples_2[:, :, field_index] - std_field[:, :, field_index]
            [std_distance[coarse_graining_index, field_index, k] = norm(Δ, kk) / norm(ones(128,128), kk) for (k,kk) in enumerate([1, 2, Inf])]
        end
    end

    tmp = return_data_file(level_index)
    α = 2e-4
    g = 9.8
    scale = [tmp.u_2std, tmp.v_2std, tmp.w_2std, tmp.b_2std / (α * g)]
    push!(level_errors, error_list) 
    push!(scales, scale)
    for k in 1:3
        fig = Figure(resolution = (1000, 1000)) 
        cglist = round.(Int, log2.(factors ))
        for i in 1:4
            ax = Axis(fig[i, 1]; title = "Prediction Error " * fieldnames[i], ylabel = units[i], xlabel)
            scatter!(ax, cglist, error_list[:, i, k] * scale[i])
            ax.xticks = cglist
        end
        for i in 1:4 
            ax = Axis(fig[i, 2]; title = "Mean Distance " * fieldnames[i], ylabel = units[i])
            scatter!(ax, cglist, mean_distance[:, i, k] * scale[i])
            ax.xticks = cglist
        end
        for i in 1:4
            ax = Axis(fig[i, 3]; title = "Standard Deviation Distance " * fieldnames[i], ylabel = units[i])
            scatter!(ax, cglist, std_distance[:, i, k] * scale[i])
            ax.xticks = cglist
        end
        save("Figures/prediction_error_level_index_$(level_index)" * "_" * normlabel[k] * ".png", fig)
    end
end


cglist = round.(Int, log2.(factors ))
colors = [:red, :blue, :green, :orange, :purple, :black]
string_label = [string(zlevel)[2:4] * " [m]" for zlevel in zlevels]
string_label[end] = "79 [m]"
string_label[1] = "1355 [m]"

for k in 1:3
    fig = Figure(resolution = (1500, 300)) 
    for i in 1:4
        ax = Axis(fig[1, i]; title = "Error " * fieldnames[i], ylabel = units[i], xlabel)
        for ii in eachindex(string_label)
            error_list = level_errors[ii]
            scale = scales[ii]
            scatter!(ax, cglist, error_list[:, i, k] * scale[i], color = colors[ii])
            lines!(ax, cglist, error_list[:, i, k] * scale[i], label = string_label[ii], color = colors[ii])
            scatter!(ax, [0, 0]; markersize = 0)
            ax.xticks = cglist
        end
        if i == 1
            axislegend(ax, position = :lt)
        end
    end

    save("Figures/prediction_error_at_various_levels_norm_" *  normlabel[k] * ".png", fig)
end


k = 2
fig = Figure(resolution = (2250, 450), fontsize = 30) 
for i in 1:4
    ax = Axis(fig[1, i]; title = "Error " * fieldnames[i], ylabel = units[i], xlabel)
    for ii in eachindex(string_label)
        error_list = level_errors[ii]
        scale = scales[ii]
        scatter!(ax, cglist, error_list[:, i, k] * scale[i], color = colors[ii])
        lines!(ax, cglist, error_list[:, i, k] * scale[i], label = string_label[ii], color = colors[ii])
        scatter!(ax, [0, 0]; markersize = 0)
        ax.xticks = cglist
    end
    if i == 2
        axislegend(ax, position = :lt, labelsize = 24)
    end
end

save("Figures/prediction_error_at_various_levels_norm_" *  normlabel[k] * ".png", fig)


fig = Figure(resolution = (1500, 3 * 300)) 
for k in 1:3
    for i in 1:4
        ax = Axis(fig[k, i]; title = normlabel[k] * " Error " * fieldnames[i], ylabel = units[i], xlabel)
        for ii in eachindex(string_label)
            error_list = level_errors[ii]
            scale = scales[ii]
            scatter!(ax, cglist, error_list[:, i, k] * scale[i], color = colors[ii])
            lines!(ax, cglist, error_list[:, i, k] * scale[i], label = string_label[ii], color = colors[ii])
            if i != 3
                scatter!(ax, [0, 0]; markersize = 0)
            end
            ax.xticks = cglist
        end
        if (i == 1) & (k == 1)
            axislegend(ax, position = :lt)
        end
    end
end
save("Figures/prediction_error_at_various_levels_norm.png", fig)
