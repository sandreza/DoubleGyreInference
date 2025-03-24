using HDF5, Statistics, ProgressBars

function coarse_grained(field, factor)
    N = size(field)[1]
    new_field = zeros(N, N)
    NN = N ÷ factor
    for i in 1:NN 
        for j in 1:NN 
            is = (i-1)*factor+1:i*factor
            js = (j-1)*factor+1:j*factor
            new_field[is, js] .= mean(field[is, js])
        end
    end
    return new_field
end

data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(data_directory_training * "eta_to_uvwb_at_z1_128_5.hdf5", "r")
eta_field = read(hfile["field"])[:, :, 5, :]
close(hfile)



new_eta_field = zeros(128, 128, 8, 4050)
for i in ProgressBar(1:size(eta_field, 3))
    for j in 0:7
        new_eta_field[:, :, j+1, i] .= coarse_grained(eta_field[:, :, i], 2^j)
    end
end

data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
hfile = h5open(data_directory_training * "eta_coarse_grained.hdf5", "w")
hfile["etas"] = Float32.(new_eta_field)
close(hfile)