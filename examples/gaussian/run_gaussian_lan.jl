###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Plots, Flux, Distributions, Random, ProgressMeter
using Flux: params
using BSON: @save
include("functions.jl")
Random.seed!(2202152)
###################################################################################################
#                                     Generate Training Data
###################################################################################################
# number of parameter vectors for training 
n_parms = 1000
# number of data points per parameter vector 
n_samples = 100
# training data
data = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
# true values 
labels = map(i -> pdf(Normal(data[1,i], data[2,i]), data[3,i]), 1:size(data,2))
labels = reshape(labels, 1, length(labels))
###################################################################################################
#                                        Create Network
###################################################################################################
# 3 nodes in input layer, 3 hidden layers, 1 node for output layer
model = Chain(
    Dense(3, 80, tanh),
    Dense(80, 80, tanh),
    Dense(80, 96, tanh),
    Dense(96, 1, identity)
)

# check our model
params(model)

# loss function
loss_fn(a, b) = Flux.mse(model(a), b) 

# optimization algorithm 
opt = ADAM(0.005)
###################################################################################################
#                                       Train Network
###################################################################################################
# number of Epochs to run
n_epochs = 500

# train the model
loss = train_model(model,n_epochs, loss_fn, data, labels, opt)

# save the model for later
@save "gaussian_model.bson" model
###################################################################################################
#                                      Plot Training
###################################################################################################
# plot the loss data
loss_plt = plot(1:n_epochs, loss, xlabel="Epochs", legend=:none, ylabel="Loss (mse)")

# plot predictions against true values 
scatter(
    labels[:], 
    model(data)[:], 
    xlabel="true density", 
    ylabel="predicted density", 
    grid=false,
    leg=false
)

# interpolated densities 
plots = Plots.Plot[]
for i in 1:20
    parms = rand_parms()
    dist = Normal(parms...)

    x = range(-parms.σ′*2 + parms.μ, parms.σ′*2 + parms.μ, length=100)
    y1 = map(x -> pdf(dist, x), x)
    y2 = mapreduce(x -> model([parms...,x]),vcat, x)

    p1 = plot(x, y1, grid=false, label="true")
    plot!(x, y2, label="predicted")
    push!(plots, p1)
    display(p1)
    sleep(.3)
end