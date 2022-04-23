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
# https://stackoverflow.com/questions/16226692/git-how-to-add-a-file-but-not-track-it/16229387
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