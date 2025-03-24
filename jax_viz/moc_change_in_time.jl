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

cg = 0
file_string = "attention_velocity_uc_production_jax_samples_"
# file_string = "old_velocity_uc_production_jax_samples_"
# file_string = "velocity_uc_production_jax_samples_"
# file_string = "regular_uc_production_jax_samples_"
# file_string = "regular_production_jax_samples_"
# file_string = "production_jax_samples_"

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
    for blev in 1:Nb
        # if blevels[blev] ≤ bmax
            for k in 1:Nz
                ψint[i, j, blev] += v[i, j, k] * dx[i] * dz[k] * (b[i, j, k] ≤ blevels[blev] )
            end
        # end
    end
    #=
    for k in 1:Nz
        if b[i, j, k] ≤ blevels[end]
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
    =#

end


MOC_data_list = [] 
MOC_samples_list  = []
MOC_mean_list = []
MOC_std_list  = []


for future_year in 1:25
    sorted_blevels_data .= Float32(0.0)
    sorted_vlevels_data .= Float32(0.0)
    sorted_blevels_samples .= Float32(0.0)
    sorted_vlevels_samples .= Float32(0.0)
    for level in ProgressBar(1:15)
        (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string = file_string, cg)
        sorted_blevels_data[:, :, level] .+= (ground_truth .* sigma .+ mu) / 25 
        sorted_blevels_samples[:, :, level, :] .+= (samples .* sigma .+ mu) / 25 
        (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string = file_string, cg)
        sorted_vlevels_data[:, :, level] .+= (ground_truth .* sigma .+ mu)  / 25
        sorted_vlevels_samples[:, :, level, :] .+= (samples .* sigma .+ mu)  / 25
    end
end

# Calculate the MOC!
dx = (dx[1:2:end] + dx[2:2:end])/2
dy = (dy[1:2:end] + dy[2:2:end])/2
dz = dz

geometric_factor = reshape(cosd.(range(15, 75, length = 128)), (128, 1))
blevels = range(extrema(sorted_blevels_data)..., length = 128 * 2)
sverdrup = 10^6
MOC_data = (calculate_residual_MOC(sorted_vlevels_data, sorted_blevels_data, dx, dy, dz; blevels=blevels) / sverdrup) .* geometric_factor 
MOC_samples = [geometric_factor .* calculate_residual_MOC(sorted_vlevels_samples[:, :, :, i], sorted_blevels_samples[:, :, :, i], dx, dy, dz; blevels=blevels) / sverdrup for i in ProgressBar(1:nsamples)] 
MOC_mean = mean(MOC_samples)
MOC_std = std(MOC_samples)

push!(MOC_data_list, MOC_data)
push!(MOC_samples_list, MOC_samples)
push!(MOC_mean_list, MOC_mean)
push!(MOC_std_list, MOC_std)

for future_year in 26:50
    sorted_blevels_data .= Float32(0.0)
    sorted_vlevels_data .= Float32(0.0)
    sorted_blevels_samples .= Float32(0.0)
    sorted_vlevels_samples .= Float32(0.0)
    for level in ProgressBar(1:15)
        (; ground_truth, samples, mu, sigma) = jax_field(level, :b, future_year; file_string = file_string, cg)
        sorted_blevels_data[:, :, level] .+= (ground_truth .* sigma .+ mu) / 25 
        sorted_blevels_samples[:, :, level, :] .+= (samples .* sigma .+ mu) / 25 
        (; ground_truth, samples, mu, sigma) = jax_field(level, :v, future_year; file_string = file_string, cg)
        sorted_vlevels_data[:, :, level] .+= (ground_truth .* sigma .+ mu)  / 25
        sorted_vlevels_samples[:, :, level, :] .+= (samples .* sigma .+ mu)  / 25
    end
end

geometric_factor = reshape(cosd.(range(15, 75, length = 128)), (128, 1))
blevels = range(extrema(sorted_blevels_data)..., length = 128 * 2)
sverdrup = 10^6
MOC_data = (calculate_residual_MOC(sorted_vlevels_data, sorted_blevels_data, dx, dy, dz; blevels=blevels) / sverdrup) .* geometric_factor 
MOC_samples = [geometric_factor .* calculate_residual_MOC(sorted_vlevels_samples[:, :, :, i], sorted_blevels_samples[:, :, :, i], dx, dy, dz; blevels=blevels) / sverdrup for i in ProgressBar(1:nsamples)] 
MOC_mean = mean(MOC_samples)
MOC_std = std(MOC_samples)

push!(MOC_data_list, MOC_data)
push!(MOC_samples_list, MOC_samples)
push!(MOC_mean_list, MOC_mean)
push!(MOC_std_list, MOC_std)

##
b_surf = max.(0, maximum(sorted_blevels_data[:, :, 15], dims=1)[1, :])

hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (6 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "Oceananigans 1-25")
cval = maximum(abs.(MOC_data))
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 30)
contourf!(ax, lat, blevels, MOC_data_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Oceananigans 26-50")
contourf!(ax, lat, blevels, MOC_data_list[2], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Difference")
cm = contourf!(ax, lat, blevels, MOC_data_list[2] -MOC_data_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 4],  cm, label = "Sverdrup [m³ s⁻¹]")

ax = Axis(fig[2, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI 1-25")
contourf!(ax, lat, blevels, MOC_mean_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[2, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI 26-50")
contourf!(ax, lat, blevels, MOC_mean_list[2], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[2, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Difference")
cm = contourf!(ax, lat, blevels, MOC_mean_list[2] -MOC_mean_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 4],  cm, label = "Sverdrup [m³ s⁻¹]")

save(pwd() * "/Figures/jax_moc_data_in_time.png", fig)


##

hiding_options = (; label = true, ticklabels = true, ticks = false, grid = false, minorgrid = false, minorticks = false)
factor = 120
fig = Figure(resolution = (8 * factor, 4 * factor))
ax = Axis(fig[1, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy [m²s⁻¹]", title = "Oceananigans 1-25")
cval = maximum(abs.(MOC_data))
lat = range(15, 75, length = 128)
contour_levels = range(-cval, cval, length = 30)
std_levels = range(0, cval/2, length = 30)
contourf!(ax, lat, blevels, MOC_data_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[1, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Oceananigans 26-50")
contourf!(ax, lat, blevels, MOC_data_list[2], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[1, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Oceananigans Change")
cm = contourf!(ax, lat, blevels, MOC_data_list[2] -MOC_data_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 4],  cm, label = "Sverdrup [m³ s⁻¹]")

ax = Axis(fig[2, 1], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI 1-25")
contourf!(ax, lat, blevels, MOC_mean_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)

ax = Axis(fig[2, 2], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI 26-50")
contourf!(ax, lat, blevels, MOC_mean_list[2], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)

ax = Axis(fig[2, 3], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Change")
cm = contourf!(ax, lat, blevels, MOC_mean_list[2] -MOC_mean_list[1], colormap = :balance, levels = contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 4],  cm, label = "Sverdrup [m³ s⁻¹]")


ax = Axis(fig[1, 5], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "Change Difference")
field1 = MOC_mean_list[2] - MOC_mean_list[1]
field2 = MOC_data_list[2] - MOC_data_list[1]
e_contour_levels = range(0, 0.5, length = 30)
cm = contourf!(ax, lat, blevels, abs.(field1 - field2), colormap = :viridis, levels = e_contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[1, 6],  cm, label = "Sverdrup [m³ s⁻¹]")

ax = Axis(fig[2, 5], xlabel = "Latitude [ᵒ]", ylabel = "Buoyancy", title = "AI Change STD")
cm = contourf!(ax, lat, blevels, std(MOC_samples_list[2] -MOC_samples_list[1]), colormap = :viridis, levels = e_contour_levels)
xlims!(extrema(lat)...)
ylims!(extrema(blevels)...)
lines!(ax, lat, b_surf, linewidth = 2, color=:black, linestyle=:dash)
hideydecorations!(ax; hiding_options...)
Colorbar(fig[2, 6],  cm, label = "Sverdrup [m³ s⁻¹]")

save(pwd() * "/Figures/jax_moc_data_in_time_with_error_std.png", fig)