using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

analysis_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/"
M = 256 
casevar = 5
α = 2e-4
g = 9.81
Nz = 15
Lz = 1800
σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2

data_directory = "/orcd/data/raffaele/001/sandre/OceananigansData/"

N1 = 1126 # start of the training data
N2 = 3645 # end of the training data

hfile = h5open(analysis_directory * "convergence_with_depth_$casevar.hdf5", "r") 
averages = read(hfile["averages"] )
close(hfile)

averages[:, :, 4] = averages[:, :, 4] ./ (α * g)
zlevels = z_centers.(1:15)

levels = [3, collect(9:14)...] # with respect to the original indices 

# fig = Figure(resolution = (1000, 750))
# state_index = 1
# state_names = ["U", "V", "W", "T"]
# units = ["m/s", "m/s", "m/s", "ᵒC"]
# end_index = (size(averages, 1) - N2 ) ÷ 2 + N2
# chosen_levels = levels[1:2:end]
# hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
# for (i, level) in enumerate(chosen_levels)
#     for state_index in eachindex(1:4)
#         depth_string = @sprintf("%0.f", abs(zlevels[level]))
#         ax = Axis(fig[state_index, length(chosen_levels) + 1 - i]; title = state_names[state_index] * " at Depth = " * depth_string * " [m]", xlabel = "Time (months)", ylabel = state_names[state_index] * " [" * units[state_index] * "]")
#         if (state_index < 4) & (i < length(chosen_levels))
#             hidedecorations!(ax; hiding_options...)
#         elseif (state_index < 4)
#             hidexdecorations!(ax; hiding_options...)
#         elseif (i < length(chosen_levels))
#             hideydecorations!(ax; hiding_options...)
#         else
#             nothing
#         end
#         lines!(ax, 1:N1-1, averages[1:N1-1, level, state_index]; color = :blue, label = "Spin Up")
#         lines!(ax, N1:N2, averages[N1:N2, level, state_index]; color = :red, label = "Training Data")
#         lines!(ax, N2+1:end_index, averages[N2+1:end_index, level, state_index]; color = :orange, label = "Test Data")
#         if (state_index == 1) | (state_index == 2)
#             ylims!(ax, 0, 0.11)
#         elseif state_index == 3
#             ylims!(ax, 0, 0.0011)
#         else
#             ylims!(ax, 0, 30)
#         end
#         if (i == 1) & (state_index == 1)
#             axislegend(ax, position = :rc)
#         end
#     end
# end
# save("Figures/convergence_with_depth.png", fig)

fig = Figure(resolution = (1000, 750))
state_index = 1
ylabels = [L"\text{U ms}^{-1}", L"\text{V ms}^{-1}", L"\text{W ms}^{-1}", L"\text{T }^\circ\text{C}"]
end_index = (size(averages, 1) - N2 ) ÷ 2 + N2
chosen_levels = levels[1:2:end]

hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
for (i, level) in enumerate(chosen_levels)
    for state_index in eachindex(1:4)
        depth_string = @sprintf("%0.f", abs(zlevels[level]))
        if state_index == 1
            if level == 1 
                title = L"\text{\textbf{Depth = 79 m}}"
            elseif level == 2
                title = L"\text{\textbf{Depth = 213 m}}"
            elseif level == 3
                title = L"\text{\textbf{Depth = 388 m}}"
            else
                title = L"\text{\textbf{Depth = 1355 m}}"
            end
        else
            title = ""
        end
        if state_index == 4
            xticks = ([0, 100, 200, 300], [L"0", L"100", L"200", L"300"])
        else
            xticks = ([0, 100, 200, 300], ["", "", "", ""])
        end

        if state_index == 1 || state_index == 2
            if length(chosen_levels) + 1 - i == 1
                yticks = ([1e-2, 10^(-1.5), 1e-1], [L"10^{-2}", L"10^{-1.5}", L"10^{-1}"])
            else
                yticks = ([1e-2, 10^(-1.5), 1e-1], ["", "", ""])
            end
        elseif state_index == 3
            if length(chosen_levels) + 1 - i == 1
                yticks = ([1e-5, 1e-4, 1e-3], [L"10^{-5}", L"10^{-4}", L"10^{-3}"])
            else
                yticks = ([1e-5, 1e-4, 1e-3], ["", "", ""])
            end
        else
            if length(chosen_levels) + 1 - i == 1
                yticks = ([3, 10, 30], [L"3", L"10", L"30"])
            else
                yticks = ([3, 10, 30], ["", "", ""])
            end
        end

        ax = Axis(fig[state_index, length(chosen_levels) + 1 - i]; 
                  title, 
                  yscale=log10, 
                  xlabel = L"\text{Time (Years)}", 
                  xticks, 
                  yticks,
                  ylabel = ylabels[state_index])

        if (state_index < 4) & (i < length(chosen_levels))
            hidedecorations!(ax; hiding_options...)
        elseif (state_index < 4)
            hidexdecorations!(ax; hiding_options...)
        elseif (i < length(chosen_levels))
            hideydecorations!(ax; hiding_options...)
        else
            nothing
        end
        lines!(ax, collect(1:N1-1)/12, averages[1:N1-1, level, state_index]; color = :blue, label = L"\text{Spin-up}")
        lines!(ax, collect(N1:N2)/12, averages[N1:N2, level, state_index]; color = :red, label = L"\text{Training data}")
        lines!(ax, collect(N2+1:end_index)/12, averages[N2+1:end_index, level, state_index]; color = :orange, label = L"\text{Test data}")
        if (state_index == 1) | (state_index == 2)
            ylims!(ax, 0.01, 0.125)
        elseif state_index == 3
            ylims!(ax, 1e-5, 1e-3)
        else
            ylims!(ax, 1.2, 30)
        end
        if (i == 1) & (state_index == 1)
            axislegend(ax, position = :rt)
        end
    end
end
save("Figures/convergence_with_depth_log.png", fig)