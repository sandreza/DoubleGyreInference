using FFTW
using Statistics: mean
using CairoMakie, SixelTerm
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

struct Spectrum{S, F}
    spec :: S
    freq :: F
end

import Base

@show "This is the file"

Base.:(+)(s::Spectrum, t::Spectrum) = Spectrum(s.spec .+ t.spec, s.freq)
Base.:(-)(s::Spectrum, t::Spectrum) = Spectrum(s.spec .- t.spec, s.freq)
Base.:(*)(s::Spectrum, t::Spectrum) = Spectrum(s.spec .* t.spec, s.freq)
Base.:(/)(s::Spectrum, t::Number)   = Spectrum(s.spec ./ t, s.freq)

Base.real(s::Spectrum) = Spectrum(real.(s.spec), s.freq)
Base.abs(s::Spectrum)  = Spectrum( abs.(s.spec), s.freq)

@inline onefunc(args...)  = 1.0
@inline hann_window(n, N) = sin(π * n / N)^2 

function power_spectrum_1d_x(var, dx; windowing=hann_window, real_valued=true, remove_mean=true)

    Nx = length(var)
    Nfx = Int64(Nx)
    
    spectra = zeros(ComplexF64, Int(Nfx/2))
    
    freqs = fftfreq(Nfx, 1.0 / dx) # 0,+ve freq,-ve freqs (lowest to highest)
    freqs = freqs[1:Int(Nfx/2)] .* 2.0 .* π
    
    windowed_var = [var[i] * windowing(i, Nfx) for i in 1:Nfx]
    fourier      = fft(windowed_var) / Nfx
    spectra[1]  += fourier[1] .* conj(fourier[1])

    for m in 2:Int(Nfx/2)
        spectra[m] += 2.0 * fourier[m] * conj(fourier[m]) # factor 2 for neg freq contribution
    end

    if remove_mean
        freqs=freqs[2:end]
        spectra=spectra[2:end]
    end

    spectrum = Spectrum(spectra, freqs)

    return real_valued ? real(spectrum) : spectrum
end

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

α = 2e-4
g = 9.81
factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

ulevels_data = zeros(128, 128, length(files), 100)
vlevels_data = zeros(128, 128, length(files), 100)
Tlevels_data = zeros(128, 128, length(files), 100)
ulevels_samples = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
Tlevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))

sample_index = 3847

levels = 1:7
for (i, level) in enumerate(levels)
    sample_tuple = return_samples_file(level, factor; complement=false)
    data_tuple = return_data_file(level; complement=false, sample_index_1=3847-100+1:3847)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_1
    
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    field[:, :, 4, :] .= field[:, :, 4, :] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    ulevels_samples[:, :, i, :] .= average_samples[:, :, 1, :]
    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    Tlevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    ulevels_data[:, :, i, :] .= field[:, :, 1, :]
    vlevels_data[:, :, i, :] .= field[:, :, 2, :]
    Tlevels_data[:, :, i, :] .= field[:, :, 4, :]
    @show extrema(ulevels_data)
    @show extrema(vlevels_data)
    @show extrema(Tlevels_data)
    
    zlevels[i] = data_tuple.zlevel
end

levels_complement = 1:8
for (i, level) in enumerate(levels_complement)
    ii = i + length(levels)

    sample_tuple = return_samples_file(level, factor; complement = true)
    data_tuple = return_data_file(level; complement = true, sample_index_1=3847-100+1:3847)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_1
    @show extrema(field[:, :, 1:4, :])
    field[:, :, 1:4, :] .= field[:, :, 1:4, :] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    field[:, :, 4, :] .= field[:, :, 4, :] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    ulevels_samples[:, :, ii, :] .= average_samples[:, :, 1, :]
    vlevels_samples[:, :, ii, :] .= average_samples[:, :, 2, :]
    Tlevels_samples[:, :, ii, :] .= average_samples[:, :, 4, :]

    ulevels_data[:, :, ii, :] .= field[:, :, 1, :]
    vlevels_data[:, :, ii, :] .= field[:, :, 2, :]
    Tlevels_data[:, :, ii, :] .= field[:, :, 4, :]
    @show extrema(ulevels_data)
    @show extrema(vlevels_data)
    @show extrema(Tlevels_data)
    
    zlevels[ii] = data_tuple.zlevel
end

permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_ulevels_data = ulevels_data[:, :, permuted_indices, :]
sorted_vlevels_data = vlevels_data[:, :, permuted_indices, :]
sorted_Tlevels_data  = Tlevels_data[:, :, permuted_indices, :]
sorted_ulevels_samples = ulevels_samples[:, :, permuted_indices, :]
sorted_vlevels_samples = vlevels_samples[:, :, permuted_indices, :]
sorted_Tlevels_samples = Tlevels_samples[:, :, permuted_indices, :]

function average_spectra(j, k, udata, vdata, dx)
    uspec  = real(power_spectrum_1d_x(udata[:, j, k, 1], dx[j])) / 100
    vspec  = real(power_spectrum_1d_x(vdata[:, j, k, 1], dx[j])) / 100
    for t in 2:100
        uspec += power_spectrum_1d_x(udata[:, j, k, t], dx[10]) / 100
        vspec += power_spectrum_1d_x(vdata[:, j, k, t], dx[10]) / 100
    end

    return uspec + vspec
end

function kinetic_spectra(j, k, t, udata, vdata, dx)
    uspec  = real(power_spectrum_1d_x(udata[:, j, k, t], dx[j]))
    vspec  = real(power_spectrum_1d_x(vdata[:, j, k, t], dx[j]))

    return uspec + vspec
end

kspec1 = Spectrum[kinetic_spectra(10,  15, t, sorted_ulevels_samples, sorted_vlevels_samples, dx) for t in 1:100]
kspec2 = Spectrum[kinetic_spectra(64,  15, t, sorted_ulevels_samples, sorted_vlevels_samples, dx) for t in 1:100]
kspec3 = Spectrum[kinetic_spectra(120, 15, t, sorted_ulevels_samples, sorted_vlevels_samples, dx) for t in 1:100]

kdata1 = kinetic_spectra(10,  15, 100, sorted_ulevels_data, sorted_vlevels_data, dx)
kdata2 = kinetic_spectra(64,  15, 100, sorted_ulevels_data, sorted_vlevels_data, dx)
kdata3 = kinetic_spectra(120, 15, 100, sorted_ulevels_data, sorted_vlevels_data, dx)

kmean1 = mean(kspec1)
kmean2 = mean(kspec2)
kmean3 = mean(kspec3)

kfull1 = zeros(length(kmean1.spec), 100)
kfull2 = zeros(length(kmean2.spec), 100)
kfull3 = zeros(length(kmean3.spec), 100)

for j in 1:100
    kfull1[:, j] .= kspec1[j].spec
    kfull2[:, j] .= kspec2[j].spec
    kfull3[:, j] .= kspec3[j].spec
end

kq11 = deepcopy(kmean1)
kq12 = deepcopy(kmean2)
kq13 = deepcopy(kmean3)

kq21 = deepcopy(kmean1)
kq22 = deepcopy(kmean2)
kq23 = deepcopy(kmean3)

kq11.spec .= [quantile(kfull1[i, :], 0.01) for i in 1:length(kmean1.spec)]
kq12.spec .= [quantile(kfull2[i, :], 0.01) for i in 1:length(kmean2.spec)]
kq13.spec .= [quantile(kfull3[i, :], 0.01) for i in 1:length(kmean3.spec)]

kq21.spec .= [quantile(kfull1[i, :], 0.99) for i in 1:length(kmean1.spec)]
kq22.spec .= [quantile(kfull2[i, :], 0.99) for i in 1:length(kmean2.spec)]
kq23.spec .= [quantile(kfull3[i, :], 0.99) for i in 1:length(kmean3.spec)]

fig = Figure()
ax = Axis(fig[1, 1]; yscale = log10, xscale = log10, xlabel = "wavenumber", ylabel = "PSD")

lines!(ax, kdata1.freq, kdata1.spec;             color = :red, linewidth = 2, linestyle = :dash, label="South data")
lines!(ax, ksamples1.freq, ksamples1.spec;       color = :red, linewidth = 2, label="South samples")
 band!(ax, ksamples1.freq, kq11.spec, kq21.spec; color = (:red, 0.5))

lines!(ax, kdata2.freq, kdata2.spec;             color = :blue, linewidth = 2, linestyle = :dash, label = "Equator data")
lines!(ax, ksamples2.freq, ksamples2.spec;       color = :blue, linewidth = 2, label="Equator samples")
 band!(ax, ksamples2.freq, kq12.spec, kq22.spec; color = (:blue, 0.5))

lines!(ax, kdata3.freq, kdata3.spec;             color = :green, linewidth = 2, linestyle = :dash, label="North data")
lines!(ax, ksamples3.freq, ksamples3.spec;       color = :green, linewidth = 2, label="North samples")
 band!(ax, ksamples3.freq, kq13.spec, kq23.spec; color = (:green, 0.5))

axislegend(ax, position=:lb)

save("Figures/kinetic_energy_spectra.png", fig)