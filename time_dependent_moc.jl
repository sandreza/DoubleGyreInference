using KernelAbstractions: @kernel, @index, CPU
using DoubleGyreInference, Statistics, LinearAlgebra, CairoMakie, Printf
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

oceananigans_data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
level_index = 1
hfile = h5open(oceananigans_data_directory  * DoubleGyreInference.return_prefix(level_index, 1)[1:end-3] * ".hdf5", "r")
total_field = read(hfile["field"])
close(hfile)

files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

sample_indices = collect(3649:2:3847)
vlevels_data = zeros(128, 128, length(files), 100)
blevels_data = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
blevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))

factor = 1
hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    sample_tuple = return_samples_file(level, factor; complement = false)
    data_tuple = return_data_file(level; complement = false, sample_index_2 = sample_indices )

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1))
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 

    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    blevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, i, :] .= field[:, :, 2, :]
    blevels_data[:, :, i, :] .= field[:, :, 4, :]

    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (i, level) in ProgressBar(enumerate(levels_complement))
    ii = i + length(levels)

    sample_tuple = return_samples_file(level, factor; complement = true)
    data_tuple = return_data_file(level; complement = true, sample_index_2 = sample_indices )

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1))
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 

    vlevels_samples[:, :, ii, :] .= average_samples[:, :, 2, :]
    blevels_samples[:, :, ii, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, ii, :] .= field[:, :, 2, :]
    blevels_data[:, :, ii, :] .= field[:, :, 4, :]
    
    zlevels[ii] = data_tuple.zlevel
end

@inline function calculate_residual_MOC(v, b, dx, dy, dz; blevels = collect(0.0:0.001:0.06))

    Nb         = length(blevels)
    Nx, Ny, Nz = size(b)

    ψ       = zeros(Nx, Ny, Nb)
    ψint    = zeros(Nx, Ny, Nb)
    ψavgint = zeros(Ny, Nb)


    _cumulate_v_velocities!(CPU(), (16, 16), (Nx, Ny))(ψint, ψ, b, v, blevels, dx, dy, dz, Nz) 

    for i in 1:Nx
       ψavgint .+= ψint[i, :, :]
    end

    return ψavgint
end

@kernel function _cumulate_v_velocities!(ψint, ψ, b, v, blevels, dx, dy, dz, Nz)
    i, j = @index(Global, NTuple)

    Nb = length(blevels)
    Δb = blevels[2] - blevels[1]

    bmax = maximum(b[i, j, :])
    bmin = minimum(b[i, j, :])
    for k in 1:Nz
        if b[i, j, k] < blevels[end]
            blev = searchsortedfirst(blevels, b[i, j, k])
            if bmin ≤ blevels[blev] ≤ bmax 
                ψ[i, j, blev] += v[i, j, k] * dx[i] * dz[k]
            end
        end
    end

    ψint[i, j, 1] = ψ[i, j, 1]
    bmax = maximum(b[i, j, :])
    for blev in 2:Nb
        if bmin ≤ blevels[blev] ≤ bmax 
            ψint[i, j, blev] = ψint[i, j, blev-1] + ψ[i, j, blev]
        end
    end

end

# Calculate the MOC!
dx = (dx[1:2:end] + dx[2:2:end])/2
dy = (dy[1:2:end] + dy[2:2:end])/2
dz = dz

permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_vlevels_data = vlevels_data[:, :, permuted_indices, :]
sorted_blevels_data = blevels_data[:, :, permuted_indices, :]
sorted_vlevels_samples = vlevels_samples[:, :, permuted_indices, :]
sorted_blevels_samples = blevels_samples[:, :, permuted_indices, :]

geometric_factor = reshape(cosd.(range(15, 75, length = 128)), (128, 1))
blevels = range(extrema(sorted_blevels_data)..., length = 128 * 2)
sverdrup = 10^6
# MOC_data = (calculate_residual_MOC(sorted_vlevels_data, sorted_blevels_data, dx, dy, dz; blevels=blevels) / sverdrup) .* geometric_factor 
MOC_samples = [geometric_factor .* calculate_residual_MOC(sorted_vlevels_data[:, :, :, i], sorted_blevels_samples[:, :, :, i], dx, dy, dz; blevels=blevels) / sverdrup for i in ProgressBar(1:100)] 
MOC_mean_1 = mean(MOC_samples[1:50])
MOC_mean_2 = mean(MOC_samples[51:100])
# MOC_std = std(MOC_samples)

b_surf = max.(0, maximum(blevels_data[:, :, 15,1], dims=1)[1, :])

hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (6 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "first 50 months")
cval = quantile(abs.(MOC_samples[1])[:], 0.99)
clampval = quantile(abs.(MOC_samples[1])[:], 0.98)
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 30)
contourf!(ax, lat, blevels, clamp.(MOC_mean_1, -clampval, clampval), colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "latter 50 months")
contourf!(ax, lat, blevels, clamp.(MOC_mean_2, -clampval, clampval), colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)


ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "Difference")
contourf!(ax, lat, blevels, MOC_mean_2 - MOC_mean_1, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)


save("Figures/MOC_data_in_time.png", fig)
