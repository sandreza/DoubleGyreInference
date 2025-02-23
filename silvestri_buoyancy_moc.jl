using KernelAbstractions: @kernel, @index, CPU
using DoubleGyreInference, Statistics, LinearAlgebra, CairoMakie, Printf
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

vlevels_data = zeros(128, 128, length(files))
blevels_data = zeros(128, 128, length(files))
vlevels_samples = zeros(128, 128, length(files), 100)
blevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    sample_tuple = return_samples_file(level, factor; complement = false)
    data_tuple = return_data_file(level; complement = false)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 

    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    blevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, i] .= field[:, :, 2]
    blevels_data[:, :, i] .= field[:, :, 4]

    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (i, level) in ProgressBar(enumerate(levels_complement))
    ii = i + length(levels)

    sample_tuple = return_samples_file(level, factor; complement = true)
    data_tuple = return_data_file(level; complement = true)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    # field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    # average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    vlevels_samples[:, :, ii, :] .= average_samples[:, :, 2, :]
    blevels_samples[:, :, ii, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, ii] .= field[:, :, 2]
    blevels_data[:, :, ii] .= field[:, :, 4]
    
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

    for k in 1:Nz
        if b[i, j, k] < blevels[end]
            blev = searchsortedfirst(blevels, b[i, j, k])
            ψ[i, j, blev] += v[i, j, k] * dx[i] * dz[k]
        end
    end

    ψint[i, j, 1] = Δb * ψ[i, j, 1]
    bmax = maximum(b[i, j, :])
    for blev in 2:Nb
        if bmax > blevels[blev]
            ψint[i, j, blev] = ψint[i, j, blev-1] + Δb * ψ[i, j, blev]
        end
    end
end

# Calculate the MOC!
dx = (dx[1:2:end] + dx[2:2:end])/2
dy = (dy[1:2:end] + dy[2:2:end])/2
dz = dz

permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_vlevels_data = vlevels_data[:, :, permuted_indices]
sorted_blevels_data = blevels_data[:, :, permuted_indices]
sorted_vlevels_samples = vlevels_samples[:, :, permuted_indices, :]
sorted_blevels_samples = blevels_samples[:, :, permuted_indices, :]

blevels = range(extrema(sorted_blevels_data)..., length = 128 * 2)
MOC_data = calculate_residual_MOC(sorted_vlevels_data, sorted_blevels_data, dx, dy, dz; blevels=blevels)
MOC_samples = [calculate_residual_MOC(sorted_vlevels_samples[:, :, :, i], sorted_blevels_samples[:, :, :, i], dx, dy, dz; blevels=blevels) for i in ProgressBar(1:100)]
MOC_mean = mean(MOC_samples)
MOC_std = std(MOC_samples)

b_surf = max.(0, maximum(blevels_data[:, :, 15], dims=1)[1, :])


hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (6 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "Oceananigans")
cval = maximum(abs.(MOC_data))
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 30)
contourf!(ax, lat, blevels, MOC_data, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Mean")
contourf!(ax, lat, blevels, MOC_mean, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Difference")
cm = contourf!(ax, lat, blevels, MOC_data -MOC_mean, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 4],  cm, label = "[m³ s⁻¹]")

ax = Axis(fig[2, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "AI Sample 1")
contourf!(ax, lat, blevels, MOC_samples[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[2, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Sample 2")
contourf!(ax, lat, blevels, MOC_samples[end], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[2, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Uncertainty")
cm = contourf!(ax, lat, blevels, MOC_std, colormap = :viridis, levels = std_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:white, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 4],  cm, label = "[m³ s⁻¹]")
save(pwd() * "/Figures/moc_data.png", fig)

##

fig = Figure()
ax = Axis(fig[1, 1]; title = "v̄")
contourf!(ax,lat, sorted_zlevels, mean(sorted_vlevels_data, dims = 1)[1, :, :], levels = range(-0.02, 0.02, length = 20), colormap = :balance)
ax = Axis(fig[1, 2]; title = "b")
contourf!(ax, lat, sorted_zlevels,  mean(sorted_blevels_data, dims = 1)[1, :, :], levels = 10)

ax = Axis(fig[2, 1]; title = "v slice mid")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[64, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[2, 2]; title = "b slice mid")
contourf!(ax, lat, sorted_zlevels,  sorted_blevels_data[64, :, :], levels = 10)

ax = Axis(fig[3, 1]; title = "v slice beg")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[16, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[3, 2]; title = "b slice beg")
contourf!(ax, lat, sorted_zlevels,  sorted_blevels_data[16, :, :], levels = 10)

ax = Axis(fig[4, 1]; title = "v slice end")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[96, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[4, 2]; title = "b slice end")
contourf!(ax, lat, sorted_zlevels,  sorted_blevels_data[96, :, :], levels = 10)

ax = Axis(fig[3, 3]; title = "v slice mbeg")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[8, :, :], levels = range(-0.4, 0.4, length = 20), colormap = :balance)
ax = Axis(fig[4, 3]; title = "b slice mbeg")
contourf!(ax, lat, sorted_zlevels,  sorted_blevels_data[8, :, :], levels = 10)

ax = Axis(fig[1, 3]; title = "v slice mend")
contourf!(ax,lat, sorted_zlevels, sorted_vlevels_data[120, :, :], levels = range(-0.1, 0.1, length = 20), colormap = :balance)
ax = Axis(fig[2, 3]; title = "b slice mend")
contourf!(ax, lat, sorted_zlevels,  sorted_blevels_data[120, :, :], levels = 10)

save("Figures/v_b_data.png", fig)
