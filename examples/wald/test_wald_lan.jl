###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Plots, Flux, Distributions, Random, ProgressMeter
using Flux: params
using BSON: @load
include("functions.jl")
Random.seed!(9958552)
@load "wald_model.bson" model
###################################################################################################
#                                      Regenerate Data
###################################################################################################
# number of parameter vectors for training 
n_parms = 10_000
# number of data points per parameter vector 
n_samples = 250
# training data
data = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
# true values 
labels = map(i -> pdf(Normal(data[1,i], data[2,i]), data[3,i]), 1:size(data,2))
labels = reshape(labels, 1, length(labels))
###################################################################################################
#                                      Plot Densities
###################################################################################################
# plot predictions against true values
idx = rand(1:size(train_y, 2), 100_000) 
sub_train_y = train_y[idx]
pred_y = model(train_x[:,idx])[:]
residual = pred_y .- sub_train_y

scatter(
    sub_train_y, 
    pred_y, 
    xlabel = "true density", 
    ylabel = "predicted density", 
    grid = false,
    leg = false
)

scatter(
    sub_train_y, 
    residual, 
    xlabel = "true density", 
    ylabel = "residual", 
    grid = false,
    leg = false
)

# interpolated densities 
plots = Plots.Plot[]
for i in 1:20
    parms = rand_parms()
    dist = Wald(;parms...)

    x = range(0.1, 2.0, length = 100)
    y1 = map(x -> pdf(dist, x), x)
    y2 = mapreduce(x -> model([vcat(parms...)...,x]), vcat, x)

    p1 = plot(x, y1, grid=false, label="true")
    plot!(x, y2, label="predicted")
    push!(plots, p1)
    display(p1)
    sleep(.3)
end