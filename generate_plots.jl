prediction_error = Vector{Float64}[]
mean_distance = Vector{Float64}[]
std_distance = Vector{Float64}[]

level_index = 1
factor = 1
include("load_data.jl")
include("preliminary_plot_data.jl")
push!(prediction_error, [norm(tupled_data.field_2[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]) for field_index in 1:4])
push!(mean_distance, [norm(mean_field[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]) for field_index in 1:4])
push!(std_distance, [norm(std_field[:, :, field_index] - tupled_data.std_samples_2[:, :, field_index])  for field_index in 1:4])

level_index = 1
factor = 8
include("load_data.jl")
include("preliminary_plot_data.jl")
push!(prediction_error, [norm(tupled_data.field_2[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]) for field_index in 1:4])
push!(mean_distance, [norm(mean_field[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]) for field_index in 1:4])
push!(std_distance, [norm(std_field[:, :, field_index] - tupled_data.std_samples_2[:, :, field_index])  for field_index in 1:4])

level_index = 1
factor = 32
include("load_data.jl")
include("preliminary_plot_data.jl")
push!(prediction_error, [norm(tupled_data.field_2[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]) for field_index in 1:4])
push!(mean_distance, [norm(mean_field[:, :, field_index] - tupled_data.averaged_samples_2[:, :, field_index]) for field_index in 1:4])
push!(std_distance, [norm(std_field[:, :, field_index] - tupled_data.std_samples_2[:, :, field_index])  for field_index in 1:4])

level_index = 3
factor = 1
include("load_data.jl")
include("preliminary_plot_data.jl")

level_index = 3
factor = 16
include("load_data.jl")
include("preliminary_plot_data.jl")

level_index = 5
factor = 1
include("load_data.jl")
include("preliminary_plot_data.jl")


level_index = 7
factor = 1
include("load_data.jl")
include("preliminary_plot_data.jl")


fig = Figure() 
fieldnames = ["U", "V", "W", "T"]
for i in 1:4
    ax = Axis(fig[i, 1]; title = "Prediction Error " * fieldnames[i])
    scatter!(ax, [1, 8, 32], [pe[i] for pe in prediction_error])
end
for i in 1:4 
    ax = Axis(fig[i, 2]; title = "Mean Distance " * fieldnames[i])
    scatter!(ax, [1, 8, 32], [md[i] for md in mean_distance])
end
for i in 1:4
    ax = Axis(fig[i, 3]; title = "Standard Deviation Distance " * fieldnames[i])
    scatter!(ax, [1, 8, 32], [sd[i] for sd in std_distance])
end
save("Figures/prediction_error.png", fig)