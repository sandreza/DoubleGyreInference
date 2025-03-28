# using https://github.com/sandreza/JaxDiffusion
from jaxdiffusion import *
from jaxdiffusion.process.sampler import Sampler
from jaxdiffusion.process.sampler import ODESampler
from jaxdiffusion.process.velocity_sampler import VelocitySampler
from jaxdiffusion.process.velocity_sampler import VelocityODESampler
from jaxdiffusion.models.unet import AttentionUNet
from jaxdiffusion.losses.velocity_loss import velocity_conditional_make_step, velocity_conditional_batch_loss_function
import h5py

data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
data_file = "eta_to_uvwb_at_z1_128_5.hdf5"
data_file = "eta_to_uvwb_at_z8_128_5_complement.hdf5"
data_file = "full_level_training_data.hdf5"
hfile = h5py.File(data_directory_training + data_file, 'r')
field = jnp.array(hfile["field"])
hfile.close()

data_file_eta = "eta_coarse_grained.hdf5"
hfile = h5py.File(data_directory_training + data_file_eta, 'r')
eta_field = jnp.array(hfile["etas"])
hfile.close()

plot_data = False # True

data_shape = field.shape[1:]
context_channels = 1

# add util for calculating the distance
data_mean = 0
data_std = 0.5
key = jr.PRNGKey(0)
key, subkey = jax.random.split(key)
random_index_1 = jax.random.randint(key, 10, 0, field.shape[0]-context_channels)
key, subkey = jax.random.split(key)
random_index_2 = jax.random.randint(key, 10, 0, field.shape[0]-context_channels)

tmp = jnp.linalg.norm(field[random_index_1, 0:(data_shape[0]-context_channels), :, :] - field[random_index_2, 0:(data_shape[0]-context_channels), :, :], axis=(2, 3))
tmp = jnp.sqrt(jnp.sum(tmp**2 , axis = 1))
sigma_max = max(tmp) * 1.4
sigma_min = 1e-2
key, subkey = jax.random.split(key)
schedule = VarianceExplodingBrownianMotion(sigma_min, sigma_max) 

# Loss Function assumes that all context channels are the last indices of the "channel" dimension
seed = 12345

"""
unet_hyperparameters = {
    "data_shape": data_shape,
    "features": [64, 64, 64, 64, 64, 64, 64],
    "downscaling_factor": 2,
    "kernel_size": 3, 
    "midblock_length": 2,
    "context_channels": context_channels, 
}
"""

unet_hyperparameters = {
    "context_channels": context_channels,
    "data_shape": data_shape,
    "features": [96, 96, 96,96],
    "downscaling_factor": 2,
    "kernel_size": [5, 3, 3, 3], 
    "beforeblock_length": 1,
    "afterblock_length": 1,
    "midblock_length": 8,
    "final_block_length": 1,
    "heads": 8, 
    "dim_head": 16,
}

key = jr.PRNGKey(seed)
key, unet_key, subkey = jax.random.split(key, 3)

model = AttentionUNet
model_filename = "new_attention_unet_cg.mo"
if os.path.exists(model_filename):
    print("Loading file " + model_filename)
    model = load(model_filename, model)
    model_hyperparameters = load_hyperparameters(model_filename)
    print("Done Loading model with hyperparameters")
    print(model_hyperparameters)
else:
    print("File " + model_filename + " does not exist. Creating UNet")
    model = model(key = key, **unet_hyperparameters)
    print("Done Creating UNet")


# Train Test Split
test_size = 0.2
dataset_size = field.shape[0]
indices = jnp.arange(dataset_size)
perm = jax.random.permutation(subkey, indices)
test_size = int(dataset_size * test_size)
train_size = dataset_size - test_size
train_indices = perm[:train_size]
test_indices = perm[train_size:]
train_indices = indices[000:3400]
test_indices = indices[3400:]
train_data = field[train_indices, :, :, :]
train_cg_eta = eta_field[train_indices, :, :, :]
test_data = field[test_indices, :, :, :]
test_cg_eta = eta_field[test_indices, :, :, :]

# batch and permutation looping
train_size = train_data.shape[0]
test_size = test_data.shape[0]
batch_size = 8
train_skip_size = train_size // batch_size
test_skip_size = test_size // batch_size

# Training
lr=1e-3/20
wcds = optax.warmup_constant_schedule(
  init_value=lr/100,
  peak_value= lr,
  warmup_steps=5000,
)
opt = optax.chain(
  optax.clip(1.0),
  optax.adamw(learning_rate=wcds),
)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
param_count = sum(x.size for x in jax.tree_util.tree_leaves(opt_state))
print(f"Number of parameters in the model: {param_count}")

model_key, train_key, test_key, loader_key, sample_key = jr.split(key, 5) 
total_value = 0
total_size = 0
test_value = 0
total_test_size = 0
losses = []
test_losses = []
epochs = 2000
for epoch in range(epochs):
    _, subkey, subkey2, subkey3, subkey4 = jax.random.split(subkey, 5)
    perm_train = jax.random.permutation(subkey, train_size)
    perm_test  = jax.random.permutation(subkey2, test_size)
    for chunk in range(train_skip_size-1):
        _, train_key = jax.random.split(train_key)
        tr_data = train_data[perm_train[chunk*batch_size:(chunk+1)*batch_size], :, :, :]
        cg_data = train_cg_eta[perm_train[chunk*batch_size:(chunk+1)*batch_size], :, :, :]
        perm = jax.random.permutation(subkey3, jnp.arange(batch_size)) # [batch_size//2:]
        perm_cg = jax.random.permutation(subkey4, jnp.arange(8))
        tr_data = tr_data.at[perm, -context_channels:, ...].multiply(0.0) # train partially unconditional distribution
        # cg_ind = epoch%4 + 2
        for iii in range(batch_size):
            tr_data = tr_data.at[perm[iii], -context_channels:, ...].add(cg_data[perm[iii], perm_cg[iii], ...])
        value, model, train_key, opt_state = velocity_conditional_make_step(
            model, context_channels, schedule, tr_data, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
    for chunk in range(test_skip_size-1):
        _, test_key = jax.random.split(test_key)
        tst_data = test_data[perm_test[chunk*batch_size:(chunk+1)*batch_size], :, :, :]
        test_value += velocity_conditional_batch_loss_function(model, context_channels, schedule, tst_data, test_key)
        total_test_size += 1
    print(f"------Epoch={epoch}------")
    print(f"Loss={total_value / total_size}")
    print(f"Test Loss={test_value / total_test_size}")
    losses.append(total_value / total_size)
    test_losses.append(test_value / total_test_size)
    total_value = 0
    total_size = 0
    test_value = 0
    total_test_size = 0
    save(model_filename, unet_hyperparameters, model)

# Figure and save figure
if True: 
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    axs.plot(losses, label='Train Loss')
    axs.plot(test_losses, label='Test Loss')
    axs.legend()
    plt.savefig("losses.png")

#log loss
if True: 
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    axs.plot(jnp.log(jnp.array(losses)), label='Train Loss')
    axs.plot(jnp.log(jnp.array(test_losses)), label='Test Loss')
    axs.legend()
    plt.savefig("log_losses.png")

# save the model
save(model_filename, unet_hyperparameters, model)

# Sampling
def precursor_context_model(model, context, t, y):
    y = jnp.concatenate((y, context), axis=0)
    return model(t, y)

# Plotting
first_indices = {}
for i in range(51):
    first_indices[i] = test_indices[12 * i]

# range(8)
for cg_ii in range(3, 8):
    for ii in range(51):
        context_ind = first_indices[ii]
        context = eta_field[context_ind, cg_ii:(cg_ii+1), :, :]
        tmp = jnp.zeros((1, 128, 128))
        context_model = ft.partial(precursor_context_model, model, context)

        # Sampling
        print("Sampling " + str(ii) + " at cg " + str(cg_ii))
        new_data_shape = field[0, 0:(data_shape[0]-1), :, :].shape
        sampler = VelocityODESampler(schedule, context_model, new_data_shape)
        sqrt_N = 10
        samples = sampler.sample(sqrt_N**2, steps = 300)
        print("Done Sampling, Now Plotting")
        # plotting
        if ii == 0:
            plot_data = True
        else:
            plot_data = False

        for k in range((data_shape[0]-1)):
            sample = jnp.reshape(samples[:, k, :, :], (sqrt_N, sqrt_N, 128, 128))

            filename = data_directory_training + 'attention_velocity_uc_production_jax_samples_' + str(ii) + '_field_' + str(k) + '_cg_' + str(cg_ii) +  '.hdf5'
            # print("Saving samples to " + filename)
            with h5py.File(filename, "w") as f:
                f.create_dataset("samples", data=samples[:, k, :, :] )
                f.create_dataset("context", data=context )
                f.create_dataset("ground_truth", data=field[context_ind, k, :, :] )
            
            # print("Done Saving samples to " + filename)
            # sample = data_mean + data_std * sample
            sample_average = jnp.mean(sample, axis = (0, 1))
            sample_std = jnp.std(sample, axis = (0, 1))
            max_sigma = jnp.max(sample_std)/2
            conditional_information = context * data_std + data_mean 

            if plot_data:
                sqrtN_plot = 4
                fig, axes = plt.subplots(sqrtN_plot, sqrtN_plot, figsize=(sqrtN_plot, sqrtN_plot))
                # Plot the original images (0 index of axis 1)
                for i in range(sqrtN_plot):
                    for j in range(sqrtN_plot):
                        if i == j == 0: 
                            axes[j, i].imshow(context[0, :, :])
                            axes[j, i].set_title(f"Context")
                            axes[j, i].axis("off")
                        elif (i == 1) & (j == 0):
                            axes[j, i].imshow(field[context_ind, k, :, :], cmap = 'bwr', vmin = -1 / 1, vmax = 1 / 1)
                            axes[j, i].set_title(f"Ground Truth")
                            axes[j, i].axis("off")
                        elif (i == 2) & (j == 0):
                            axes[j, i].imshow(sample_average, cmap = 'bwr', vmin = -1 / 1, vmax = 1 / 1)
                            axes[j, i].set_title(f"Ensemble Average")
                            axes[j, i].axis("off")
                        elif (i == 3) & (j == 0):
                            axes[j, i].imshow(sample_average - field[context_ind, k, :, :], cmap = 'bwr', vmin = -1 / 1, vmax = 1 / 1)
                            axes[j, i].set_title(f"Difference")
                            axes[j, i].axis("off")
                        elif (i == 3) & (j == 1):
                            axes[j, i].imshow(sample_std, vmin = 0, vmax = 0.5 / 1)
                            axes[j, i].set_title(f"STD")
                            axes[j, i].axis("off")
                        else:
                            axes[j, i].imshow(sample[i, j, :, :], cmap = 'bwr', vmin = -1 / 1, vmax = 1 / 1)
                            axes[j, i].set_title(f"{i}, {j}")
                            axes[j, i].axis("off")
                plt.tight_layout()
                plt.show()
                filename = "velocity_production_" + str(ii) + "_field_" + str(k) + '_cg_' + str(cg_ii) + "_ode.png"
                plt.savefig(filename)