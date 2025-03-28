using HDF5, CairoMakie

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dz = read(hfile["dz"])
close(hfile)

data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
data_file = "full_level_training_data.hdf5"
hfile = h5open(data_directory_training * data_file, "r")
field = read(hfile["field"])
mus = read(hfile["mus"])
stds = read(hfile["stds"])
close(hfile)


field = field .* reshape(stds, (1, 1, 61, 1)) .+ reshape(mus, (1, 1, 61, 1))


vfield = field[:, :, 2:4:end, :];
tfield = field[:, :, 4:4:end, :];

hflux = sum(vfield .* tfield .* reshape(dz, (1, 1, 15, 1)), dims = (1, 3))

fig = Figure() 
ax = Axis(fig[1, 1], title = "Heat Flux in Time", xlabel = "Time", ylabel = "Heat Flux")
lines!(ax, hflux[1, :, 1,  1], label = "Heat Flux", color = :blue)
ax = Axis(fig[1, 2], title = "Heat Flux in Time", xlabel = "Time", ylabel = "Heat Flux")
lines!(ax, hflux[1, :, 1,  end], label = "Heat Flux", color = :blue)
save("heat_flux_in_time.png", fig)


# save hflux 
hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/heat_flux_in_time.hdf5", "w")
hfile["heat_flux"] = hflux[1, :, 1, :]
close(hfile)