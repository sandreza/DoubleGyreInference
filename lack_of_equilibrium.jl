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

fig = Figure(resolution = (2000, 1000))
state_index = 1
state_names = ["U", "V", "W", "T"]
units = ["m/s", "m/s", "m/s", "K"]
for i in eachindex(levels) 
    for state_index in eachindex(1:4)
        level = levels[i]
        depth_string = @sprintf("%0.f", abs(zlevels[level]))
        ax = Axis(fig[state_index, 8 - i]; title = state_names[state_index] * " at Depth = " * depth_string * " [m]", xlabel = "Time (months)", ylabel = state_names[state_index] * " [" * units[state_index] * "]")
        lines!(ax, 1:N1-1, averages[1:N1-1, level, state_index]; color = :blue, label = "Spin Up")
        lines!(ax, N1:N2, averages[N1:N2, level, state_index]; color = :red, label = "Training Data")
        lines!(ax, N2+1:size(averages, 1), averages[N2+1:end, level, state_index]; color = :orange, label = "Test Data")
        if (state_index == 1) | (state_index == 2)
            ylims!(ax, 0, 0.11)
        elseif state_index == 3
            ylims!(ax, 0, 0.0011)
        else
            ylims!(ax, 0, 30)
        end
        if (i == 1) & (state_index == 1)
            axislegend(ax, position = :rc)
        end
    end
end
save("Figures/convergence_with_depth.png", fig)