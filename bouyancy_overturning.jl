using KernelAbstractions: @kernel, @index
using DoubleGyreInference, Statistics, LinearAlgebra, CairoMakie, Printf
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

vlevels_data = zeros(128, 128, length(files))
wlevel_data = zeros(128, 128, length(files))
wlevels_samples = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))

levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    sample_tuple = return_samples_file(level, factor; complement = false)
    data_tuple = return_data_file(level; complement = false)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    
    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    wlevels_samples[:, :, i, :] .= average_samples[:, :, 3, :]

    vlevels_data[:, :, i] .= field[:, :, 2]
    wlevel_data[:, :, i] .= field[:, :, 3]

    zlevels[i] = data_tuple.zlevel
end

@inline function calculate_residual_MOC(v, b; blevels = collect(0.0:0.001:0.06))

    grid = v.grid
    arch = architecture(grid)

    Nb         = length(blevels)
    Nx, Ny, Nz = size(grid)
    Nt         = length(v.times) 
    
    ψ       = [zeros(Nx, Ny, Nb) for iter in 1:Nt]
    ψint    = [zeros(Nx, Ny, Nb) for iter in 1:Nt]
    ψavgint = zeros(Ny, Nb)

    for iter in 1:Nt
        @info "time $iter of $(length(v.times))"
        launch!(arch, grid, :xy, _cumulate_v_velocities!, ψint[iter], ψ[iter], b[iter], v[iter], blevels, grid, Nz)
    end

    for iter in 1:Nt
        for i in 20:220
            ψavgint .+= ψint[iter][i, :, :] / Nt
        end
    end

    return ψavgint
end

@kernel function _cumulate_v_velocities!(ψint, ψ, b, v, blevels, grid, Nz)
    i, j = @index(Global, NTuple)

    Nb = length(blevels)
    Δb = blevels[2] - blevels[1]

    @unroll for k in 1:Nz
        if b[i, j, k] < blevels[end]
            blev = searchsortedfirst(blevels, b[i, j, k])
            ψ[i, j, blev] += v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid) 
        end
    end

    ψint[i, j, 1] = Δb * ψ[i, j, 1]
    bmax = maximum(b[i, j, :])
    @unroll for blev in 2:Nb
        if bmax > blevels[blev]
            ψint[i, j, blev] = ψint[i, j, blev-1] + Δb * ψ[i, j, blev]
        end
    end

end

@inline function linear_interpolate(x, y, x₀)
    i₁ = searchsortedfirst(x, x₀)
    i₂ =  searchsortedlast(x, x₀)

    @inbounds y₂ = y[i₂]
    @inbounds y₁ = y[i₁]

    @inbounds x₂ = x[i₂]
    @inbounds x₁ = x[i₁]

    if i₁ > length(x)
        return y₂
    elseif i₁ == i₂
        isnan(y₁) && @show i₁, i₂, x₁, x₂, y₁, y₂
        return 
    else
        if isnan(y₁) || isnan(y₂) || isnan(x₁) || isnan(x₂) 
            @show i₁, i₂, x₁, x₂, y₁, y₂
        end
        return (y₂ - y₁) / (x₂ - x₁) * (x₀ - x₁) + y₁
    end
end