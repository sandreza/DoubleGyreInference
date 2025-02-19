using KernelAbstractions: @kernel, @index, CPU
using DoubleGyreInference, Statistics, LinearAlgebra, CairoMakie, Printf
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

vlevels_data = zeros(128, 128, length(files), 2401)
blevels_data = zeros(128, 128, length(files), 2401)
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
    data_tuple = return_data_file(level; complement = false, sample_index_1 = 1200:3600)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_1
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    
    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    blevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, i, :] .= field[:, :, 2, :]
    blevels_data[:, :, i, :] .= field[:, :, 4, :]

    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (ii, level) in ProgressBar(enumerate(levels_complement))
    i = ii + length(levels)
    sample_tuple = return_samples_file(level, factor; complement = true)
    data_tuple = return_data_file(level; complement = true, sample_index_1 = 1200:3600)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_1
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1))

    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    blevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    vlevels_data[:, :, i, :] .= field[:, :, 2, :]
    blevels_data[:, :, i, :] .= field[:, :, 4, :]

    zlevels[i] = data_tuple.zlevel
end

permuted_indices = sortperm(zlevels)
zlevels = zlevels[permuted_indices]
vlevels_data = vlevels_data[:, :, permuted_indices, :]
blevels_data = blevels_data[:, :, permuted_indices, :]

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

Ns = length(blevels_data[1, 1, 1, :])
blevels = collect(0:0.002:0.058)

# Calculate the MOC!
MOC = calculate_residual_MOC(vlevels_data[:, :, :, 1], blevels_data[:, :, :, 1], dx, dy, dz; blevels) ./ Ns

for i in 2:Ns
    MOC .+= calculate_residual_MOC(vlevels_data[:, :, :, i], blevels_data[:, :, :, i], dx, dy, dz; blevels) ./ Ns
end

using Statistics: mean

b_max = zeros(128)
for i in 1:Ns
    b_max .= max.(b_max, maximum(blevels_data[:, :, 15, i], dims=1)[1, :])
end

b_mean = zeros(128)
for i in 1:Ns
    b_mean .+= mean(blevels_data[:, :, 15, i], dims=1)[1, :] ./ Ns
end

fig = Figure()
ax  = Axis(fig[1, 1], title=L"\text{Overturning Circulation}", 
           xlabel=L"\text{Buoyancy ms}^{-2}", 
           ylabel=L"\text{Latitude }^\circ",
           xticks=([15, 35, 55, 75], [L"15", L"35", L"55", L"75"]),
           yticks=([0, 0.015, 0.030, 0.055], [L"0", L"0.015", L"0.030", L"0.055"]))
heatmap!(ax, range(15, 75, length=128), blevels, MOC, colorrange=(-3000, 5000), colormap=:bwr)
lines!(ax, range(15, 75, length=128), b_max, linewidth = 2, color=:black, linestyle=:dash)
lines!(ax, range(15, 75, length=128), b_mean, linewidth = 2, color=:black, linestyle=:dash)


