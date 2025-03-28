using KernelAbstractions: @kernel, @index, CPU
using DoubleGyreInference, Statistics, LinearAlgebra, CairoMakie, Printf
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

α = 2e-4
g = 9.81

total_levels =Nz = 15
Lz = 1800
nsamples = 100
sorted_vlevels_data = zeros(128, 128, total_levels)
sorted_blevels_data  = zeros(128, 128, total_levels)
sorted_blevels_samples = zeros(128, 128, total_levels, nsamples)
sorted_vlevels_samples = zeros(128, 128, total_levels, nsamples)

future_year = 50
cg = 0 # coarse_graining level
file_string = "attention_velocity_uc_production_jax_samples_"

σ = 1.3
z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_centers(k) = (z_faces(k) + z_faces(k+1) ) / 2
zlevels = z_centers.(1:Nz)
sorted_zlevels = zlevels

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

hfile = h5open("/orcd/data/raffaele/001/sandre/OceananigansData/test_vb.hdf5", "r")
fullv = read(hfile["v"] )
fullb = read(hfile["b"] )
full_dx = read(hfile["dx"])
full_dz = read(hfile["dz"])
close(hfile)


for level in ProgressBar(1:15)
    (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string, cg)
    sorted_blevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_blevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 
    (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string, cg)
    sorted_vlevels_data[:, :, level] .= (ground_truth .* sigma .+ mu) 
    sorted_vlevels_samples[:, :, level, :] .= (samples .* sigma .+ mu) 
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

# for debugging
@inline function calculate_ψ(v, b, dx, dy, dz; blevels = collect(0.0:0.001:0.06))

    Nb         = length(blevels)
    Nx, Ny, Nz = size(b)

    ψ       = zeros(Nx, Ny, Nb)
    ψint    = zeros(Nx, Ny, Nb)
    ψavgint = zeros(Ny, Nb)


    _cumulate_v_velocities!(CPU(), (16, 16), (Nx, Ny))(ψint, ψ, b, v, blevels, dx, dy, dz, Nz) 

    for i in 1:Nx
       ψavgint .+= ψ[i, :, :]
    end

    return ψavgint
end

@kernel function _cumulate_v_velocities!(ψint, ψ, b, v, blevels, dx, dy, dz, Nz)
    i, j = @index(Global, NTuple)

    Nb = length(blevels)
    Δb = blevels[2] - blevels[1]

    bmax = maximum(b[i, j, :])
    bmin = minimum(b[i, j, :])
    for blev in 1:Nb
        if (blevels[blev] ≤ bmax) & (bmin ≤ blevels[blev])
            for k in 1:Nz
                ψint[i, j, blev] += v[i, j, k] * dx[i] * dz[k] * (b[i, j, k] ≤ blevels[blev] ) # multiply by indicator function
            end
        end
    end
    #=
    for k in 1:Nz
        blev = searchsortedfirst(blevels, b[i, j, k])
        ψ[i, j, blev] += v[i, j, k] * dx[i] * dz[k] * (blevels[blev] ≤ bmax)
    end

    ψint[i, j, :] .= ψ[i, j, :]
    =#
    #=
    for blev in 2:Nb
        if bmin ≤ blevels[blev] ≤ bmax 
            ψint[i, j, blev] = ψint[i, j, blev-1] + ψ[i, j, blev]
        end
    end
    =#
end

# Calculate the MOC!
dx = (dx[1:2:end] + dx[2:2:end])/2
dy = (dy[1:2:end] + dy[2:2:end])/2
dz = dz

geometric_factor = reshape(cosd.(range(15, 75, length = 128)), (128, 1)) .* 0 .+ 1 # not needed since include in dx
blevels = range(0, 30 * α * g, length = 128 * 2)
sverdrup = 10^6
tmp = copy(sorted_blevels_data)
tmp .= mean(sorted_blevels_data, dims = 1)
maxb = [maximum(sorted_blevels_data[i,j,:]) for i in 1:128, j in 1:128]
MOC_data = (calculate_residual_MOC(sorted_vlevels_data, sorted_blevels_data, dx, dy, dz; blevels=blevels) / sverdrup) .* geometric_factor 
MOC_samples = [geometric_factor .* calculate_residual_MOC(sorted_vlevels_samples[:, :, :, i], sorted_blevels_samples[:, :, :, i], dx, dy, dz; blevels=blevels) / sverdrup for i in ProgressBar(1:nsamples)] 
MOC_mean = mean(MOC_samples)
MOC_std = std(MOC_samples)
MOC_b_data = copy(MOC_data)
##

fullv_avg = (fullv[:, 1:end-1, : ] + fullv[:, 2:end, : ])/2
full_MOC = calculate_residual_MOC(fullv_avg, fullb, full_dx, dy, full_dz; blevels=blevels)
fig = Figure() 
ax = Axis(fig[1,1])
contourf!(ax, full_MOC / 1e8, colormap = :balance, levels = range(-1, 1, length = 30))
save(pwd() * "/Figures/jax_moc_check.png", fig)

strat = fullb[:, :, 2:end] - fullb[:, :, 1:end-1]
fig = Figure(resolution = (1400, 1600)) 
for i in 1:14
    ii = (i-1) ÷ 4 + 1
    jj = (i-1) % 4 + 1
    ax = Axis(fig[ii, jj], title = "Depth $(i)")
    cval = maximum(strat[:, :, i])
    heatmap!(ax, strat[:, :, i], colormap = :balance, colorrange =(-cval, cval) )
end
save(pwd() * "/Figures/stratification.png", fig)


fig = Figure(resolution = (1400, 1600)) 
newlat = range(0, 60, length = 256)
for i in 1:16
    ii = (i-1) ÷ 4 + 1
    jj = (i-1) % 4 + 1
    ax = Axis(fig[ii, jj], title = "lat $(newlat[(i-1) * 8 + 1])")
    field = fullb[:, (i-1) * 8 + 1, :]
    cval = maximum(field)
    contourf!(ax, newlat,  zlevels, field, colormap = :thermometer, levels = range(0, cval, length = 30))
end
save(pwd() * "/Figures/bslice.png", fig)

fig = Figure(resolution = (1400, 1600)) 
newlat = range(0, 60, length = 256)
for i in 1:16
    ii = (i-1) ÷ 4 + 1
    jj = (i-1) % 4 + 1
    ax = Axis(fig[ii, jj], title = "lat $(newlat[220])")
    field = fullb[:, 220, :]
    cval = maximum(field)
    mask = field .<= (cval /16 * i)
    heatmap!(ax, newlat,  zlevels,  mask, colormap = :thermometer, colorrange = (0, 1))
end
save(pwd() * "/Figures/bslice_2.png", fig)

fig = Figure(resolution = (1400, 1600)) 
newlat = range(0, 60, length = 256)
N = 10
for i in 1:N^2
    ii = (i-1) ÷ N + 1
    jj = (i-1) % N + 1

    field = fullb[:, 220, :]
    cval = maximum(field)
    mask = (field .<= (cval /N^2 * i)) & reshape(((cval /N^2 * i) .< = maxb[: , 220]), (128, 1))
    masked_field = mask .* fullv_avg[:, 220, :] 
    avgval = masked_field .* reshape(full_dx, (256, 1)) .* reshape(full_dz, (1, 15))
    ax = Axis(fig[ii, jj], title = "$(mean(avgval))")
    heatmap!(ax, newlat,  zlevels,  masked_field, colormap = :balance, colorrange = (-0.01, 0.01))
end
save(pwd() * "/Figures/masked_v.png", fig)

##
#=
blevels = range(0, 30 * α * g, length = 10)
weighted_v = fullv_avg .* reshape(full_dx, (256, 1,  1)) .* reshape(full_dz, (1, 1, 15))
for i in eachindex(blevel)
    for j in 1:256 
        mask = (fullb .< blevels[i]) & (blevels[i] .< = maxb[i ,j])
        masked_field = mask .* fullv_avg[:, 220, :] 
    end
end
=#
##
b_surf = max.(0, maximum(sorted_blevels_data[:, :, 15], dims=1)[1, :])
b_surf_2 = [minimum(maxb[:, j]) for j in 1:128]
bmask = (reshape(b_surf, (128, 1)) .>= reshape(blevels, (1, 256))) .* 0 .+1
hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (6 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "Oceananigans")
cval = maximum(abs.(MOC_data)) 
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 30)
contourf!(ax, lat, blevels, MOC_data .* bmask, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
lines!(ax, lat, b_surf_2, linewidth = 2, color=:black, linestyle=:dot)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Mean")
contourf!(ax, lat, blevels, MOC_mean  .* bmask, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Difference")
cm = contourf!(ax, lat, blevels, (MOC_data -MOC_mean)  .* bmask, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
lines!(ax, lat, b_surf_2, linewidth = 2, color=:black, linestyle=:dot)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 4],  cm, label = "Sverdrup [m³ s⁻¹]")

ax = Axis(fig[2, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "AI Sample 1")
contourf!(ax, lat, blevels, MOC_samples[1]  .* bmask, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
lines!(ax, lat, b_surf_2, linewidth = 2, color=:black, linestyle=:dot)

ax = Axis(fig[2, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Sample 2")
contourf!(ax, lat, blevels, MOC_samples[end]  .* bmask, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
lines!(ax, lat, b_surf_2, linewidth = 2, color=:black, linestyle=:dot)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[2, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Uncertainty")
cm = contourf!(ax, lat, blevels, MOC_std  .* bmask, colormap = :viridis, levels = std_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:white, linestyle=:dash)
lines!(ax, lat, b_surf_2, linewidth = 2, color=:black, linestyle=:dot)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 4],  cm, label = "Sverdrup [m³ s⁻¹]")
save(pwd() * "/Figures/jax_moc_data_$(future_year).png", fig)

fig = Figure() 
blevels = range(extrema(sorted_blevels_data)..., length = 128 * 2)
ax = Axis(fig[1, 1], ylabel = "Latitude [ᵒ]", xlabel = "Buoyancy [m²s⁻¹]", title = "Oceananigans")
lines!(ax, sum(-MOC_b_data  .* bmask, dims = 2)[:], lat)
save(pwd() * "/Figures/jax_integrated_moc_$(future_year).png", fig)


##
#=
v̄_data = mean(sorted_vlevels_data, dims = 1)[1, :, :]
v̄_samples = mean(sorted_vlevels_samples, dims = (1, 4))[1, :, :, 1]

rad_to_deg = π / 180
dλ = 60 / 128 * rad_to_deg
Lλ = 60 * rad_to_deg
R_earth = 40007863 / (2π)
vfactor = reshape(cosd.(range(15, 75, length = 128)), (128, 1)) * R_earth / 1e6  * Lλ 
Δz = reshape(dz, (1, 15))

Ψᵛ_data = reverse(vfactor .* cumsum(reverse(v̄_data .* Δz, dims = 2), dims = 2), dims = 2)
Ψᵛ_samples = reverse(vfactor .* cumsum(reverse(v̄_samples .* Δz, dims = 2), dims = 2), dims = 2)
Ψᵛ_sample_members = reverse(vfactor .* cumsum(reverse(mean(sorted_vlevels_samples, dims = 1) .* reshape(Δz, (1, 1, 15, 1)), dims = 3), dims = 3)[1, :, :, :], dims = 2)
Ψᵛ_sample_std = reverse(vfactor .* std(cumsum(reverse(mean(sorted_vlevels_samples, dims = 1) .* reshape(Δz, (1, 1, 15, 1)), dims = 3), dims = 3), dims = 4)[1, :, :, 1], dims = 2)

blevels = sorted_zlevels
MOC_data = Ψᵛ_data
MOC_mean = Ψᵛ_samples
MOC_samples = Ψᵛ_sample_members
MOC_std = Ψᵛ_sample_std
hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (6 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "Oceananigans")
cval = maximum(abs.(MOC_data))
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 10)
contourf!(ax, lat, blevels, MOC_data, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Mean")
contourf!(ax, lat, blevels, MOC_mean, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "Difference")
cm = contourf!(ax, lat, blevels, MOC_data -MOC_mean, colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 4],  cm, label = "Sverdrup [m³ s⁻¹]")

ax = Axis(fig[2, 1], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Sample 1")
contourf!(ax, lat, blevels, MOC_samples[:, :, 1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[2, 2], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Sample 2")
contourf!(ax, lat, blevels, MOC_samples[:, :, end], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
ax = Axis(fig[2, 3], xlabel = "Latitude [ᵒ]", ylabel = "Depth [m]", title = "AI Uncertainty")
cm = contourf!(ax, lat, blevels, MOC_std, colormap = :viridis, levels = std_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
# lines!(ax, lat, b_surf, linewidth = 2, color=:white, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 4],  cm, label = "Sverdrup [m³ s⁻¹]")
save(pwd() * "/Figures/jax_v_moc_physical_$(future_year).png", fig)
=#