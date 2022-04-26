###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Plots, Flux, Distributions, Random, ProgressMeter, SequentialSamplingModels
using Flux: params
using BSON: @save
include("functions.jl")
Random.seed!(858532)
###################################################################################################
#                                     Generate Training Data
###################################################################################################
# number of parameter vectors for training 
n_parms = 2_000
# number of data points per parameter vector 
n_samples = 250
# training data
data = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
# true values 
labels = map(i -> gen_label(data[:,i]), 1:size(data,2))
labels = reshape(labels, 1, length(labels))
all_data = Flux.Data.DataLoader((data, labels), batchsize=1000)
###################################################################################################
#                                        Create Network
###################################################################################################
# 7 nodes in input layer, 3 hidden layers, 1 node for output layer
model = Chain(
    Dense(7, 100, tanh),
    #BatchNorm(100, relu),
    Dense(100, 100, tanh),
    #BatchNorm(100, relu),
    Dense(100, 120, tanh),
    #BatchNorm(120, relu),
    Dense(120, 1, identity)
)

# check our model
params(model)

# loss function
loss_fn(a, b) = Flux.huber_loss(model(a), b) 

# optimization algorithm 
opt = ADAM(0.001, (.7,.7))
###################################################################################################
#                                       Train Network
###################################################################################################
# number of Epochs to run
n_epochs = 50

# train the model
loss = train_model(model, n_epochs, loss_fn, all_data, opt)

# save the model for later
@save "lba_model.bson" model
###################################################################################################
#                                      Plot Training
###################################################################################################
# plot the loss data
loss_plt = plot(1:n_epochs, loss, xlabel="Epochs", legend=:none, ylabel="Loss (huber)")

# plot predictions against true values 
scatter(
    labels[:], 
    model(data)[:], 
    xlabel = "true density", 
    ylabel = "predicted density", 
    grid = false,
    leg = false
)