using FFTW
using Statistics: mean
using CairoMakie, SixelTerm
using DoubleGyreInference, Statistics, ProgressBars, LinearAlgebra, CairoMakie, Printf, HDF5

struct Spectrum{S, F}
    spec :: S
    freq :: F
end

import Base

Base.:(+)(s::Spectrum, t::Spectrum) = Spectrum(s.spec .+ t.spec, s.freq)
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


α = 2e-4
g = 9.81
factor = 1
files = filter(x -> endswith(x, "1_generative_samples.hdf5"), readdir("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"))

ulevels_data = zeros(128, 128, length(files))
vlevels_data = zeros(128, 128, length(files))
Tlevels_data = zeros(128, 128, length(files))
ulevels_samples = zeros(128, 128, length(files), 100)
vlevels_samples = zeros(128, 128, length(files), 100)
Tlevels_samples = zeros(128, 128, length(files), 100)
zlevels = zeros(length(files))

levels = 1:7
for (i, level) in ProgressBar(enumerate(levels))
    sample_tuple = return_samples_file(level, factor; complement = false)
    data_tuple = return_data_file(level; complement = false)

    μ, σ = return_scale(data_tuple)
    field = data_tuple.field_2
    field[:, :, 1:4] .= field[:, :, 1:4] .* reshape(σ, (1, 1, 4)) .+ reshape(μ, (1, 1, 4))
    field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    ulevels_samples[:, :, i, :] .= average_samples[:, :, 1, :]
    vlevels_samples[:, :, i, :] .= average_samples[:, :, 2, :]
    Tlevels_samples[:, :, i, :] .= average_samples[:, :, 4, :]

    ulevels_data[:, :, i] .= field[:, :, 1]
    vlevels_data[:, :, i] .= field[:, :, 2]
    Tlevels_data[:, :, i] .= field[:, :, 4]

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
    field[:, :, 4] .= field[:, :, 4] ./ (α * g)
    average_samples = sample_tuple.samples_2 .* reshape(σ, (1, 1, 4, 1)) .+ reshape(μ, (1, 1, 4, 1)) 
    average_samples[:, :, 4, :] .= average_samples[:, :, 4, :] ./ (α * g)

    ulevels_samples[:, :, ii, :] .= average_samples[:, :, 1, :]
    vlevels_samples[:, :, ii, :] .= average_samples[:, :, 2, :]
    Tlevels_samples[:, :, ii, :] .= average_samples[:, :, 4, :]

    ulevels_data[:, :, ii] .= field[:, :, 1]
    vlevels_data[:, :, ii] .= field[:, :, 2]
    Tlevels_data[:, :, ii] .= field[:, :, 4]
    
    zlevels[ii] = data_tuple.zlevel
end

hfile = h5open("/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/moc_5.hdf5", "r")
dx = read(hfile["dx"])
dy = read(hfile["dy"])
dz = read(hfile["dz"])
close(hfile)

permuted_indices = sortperm(zlevels)
sorted_zlevels = zlevels[permuted_indices]
sorted_ulevels_data = ulevels_data[:, :, permuted_indices]
sorted_vlevels_data = vlevels_data[:, :, permuted_indices]
sorted_Tlevel_data  = Tlevels_data[:, :, permuted_indices]
sorted_ulevels_samples = ulevels_samples[:, :, permuted_indices, :]
sorted_vlevels_samples = vlevels_samples[:, :, permuted_indices, :]
sorted_Tlevels_samples = Tlevels_samples[:, :, permuted_indices, :]

Ni = 5
spectra = real(power_spectrum_1d_x(sorted_ulevels_data[:, 1, 15], dx[1]; windowing=hann_window)) / Ni
for j in 2:5
    spectra += power_spectrum_1d_x(sorted_ulevels_data[:, j, 15], dx; windowing=hann_window) / Ni
end

