using HDF5, CairoMakie, Statistics, Random

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/heat_flux_in_time.hdf5", "r")
hflux = read(hfile["heat_flux"])
close(hfile)





fig = Figure(resolution = (400, 300))
ax = Axis(fig[1, 1]; title = "Zonal and Vertically Integrated Meridional Heat Flux",  ylabel = "Latitude [ᵒ]", xlabel = "Heat Flux [PW]")
latitude = range(15, 75, length = 128)
Nlat = length(latitude)
for qu in [0.6, 0.7, 0.8, 0.9]
    δlower = [quantile(hflux[i, :], 1-qu) for i in 1:Nlat]
    δupper = [quantile(hflux[i, :], qu) for i in 1:Nlat]
    band!(ax, Point.(δlower, latitude), Point.(δupper, latitude); color = (:red, 0.2))
end
save("heat_flux_quantiles.png", fig)