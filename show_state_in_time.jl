using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

M = 256 
casevar = 5
α = 2e-4
g = 9.81
Nz = 15
Lz = 1800
σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2

level_index = 3
M = 128 
case_var = 5
factor = 1
data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(data_directory_training * "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar).hdf5", "r")
field = read(hfile["field"])
close(hfile)


fig = Figure(resolution = (2000, 1000))
for state_index in 1:5
    for (i, month) in enumerate([1000, 1001, 2000, 3000, 3645, 4000])
        ax = Axis(fig[state_index, i]; title = "State = " * string(state_index) * " at Time = " * string(month), xlabel = "Longitude", ylabel = "Latitude")
        heatmap!(ax, field[:, :, state_index, month], colormap = :balance, colorrange = (-1, 1))
    end
end
save("Figures/show_state_in_time_levelindex_$(level_index).png", fig)