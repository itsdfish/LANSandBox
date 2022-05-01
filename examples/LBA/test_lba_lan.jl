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
Random.seed!(858532)
@load "lba_model.bson" model
###################################################################################################
#                                      Regenerate Data
###################################################################################################
# number of parameter vectors for training 
n_parms = 60_000
# number of data points per parameter vector 
n_samples = 200
# training data
train_x = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
train_x = Float32.(train_x)
# true values 
train_y = map(i -> gen_label(train_x[:,i]), 1:size(train_x, 2))
train_y = Float32.(train_y)
train_y = reshape(train_y, 1, length(train_y))
###################################################################################################
#                                      Plot Densities
###################################################################################################
# plot predictions against true values
idx = rand(1:size(train_y, 2), 100_000) 
sub_train_y = train_y[idx]
pred_y = model(train_x[:,:idx])[:]
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
    dist = LBA(;parms...)

    x = range(0.1, 2.0, length=200)
    y1 = map(x -> pdf(dist, 1, x), x)
    y2 = mapreduce(x -> model([vcat(parms...)...,1,x]), vcat, x)

    p1 = plot(x, y1, grid=false, label="true")
    plot!(x, y2, label="predicted")
    push!(plots, p1)
    display(p1)
    sleep(.3)
end