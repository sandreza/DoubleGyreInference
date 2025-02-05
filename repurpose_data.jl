using DoubleGyreInference, HDF5

level = 7
sf = return_samples_file(level, 1)
df = return_data_file(level)

hfile = h5open("starting_dataset.hdf5", "w")
hfile["context"] = sf.context_field_2[:, :, 1, 1]
hfile["samples"] = sf.samples_2
hfile["ground truth"] = df.field_2[:, :, 1:4]
close(hfile)

