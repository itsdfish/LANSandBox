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
n_parms = 25_000
# number of data points per parameter vector 
n_samples = 250
# training data
train_x = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
# true values 
train_y = map(i -> pdf(Normal(train_x[1,i], train_x[2,i]), train_x[3,i]), 1:size(train_x,2))
train_y = reshape(train_y, 1, length(train_y))
train_data = Flux.Data.DataLoader((train_x, train_y), batchsize=1000)
###################################################################################################
#                                        Create Network
###################################################################################################
# 3 nodes in input layer, 3 hidden layers, 1 node for output layer
model = Chain(
    Dense(3, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 120, tanh),
    Dense(120, 1, identity)
)

# check our model
params(model)

# loss function
loss_fn(a, b) = Flux.huber_loss(model(a), b) 

# optimization algorithm 
opt = ADAM(0.002)
###################################################################################################
#                                       Train Network
###################################################################################################
# number of Epochs to run
n_epochs = 50

# train the model
train_loss,test_loss = train_model(
    model, 
    n_epochs, 
    loss_fn, 
    train_data,
    train_x,
    train_y,
    test_data, 
    opt
)

# save the model for later
@save "gaussian_model.bson" model
###################################################################################################
#                                      Plot Training
###################################################################################################
# plot the loss data
loss_plt = plot(1:n_epochs, train_loss, xlabel="Epochs", ylabel="Loss (huber)", label="training")
plot!(1:n_epochs, test_loss, label="test")

# plot predictions against true values 
scatter(
    train_y[:], 
    model(train_x)[:], 
    xlabel = "true density", 
    ylabel = "predicted density", 
    grid = false,
    leg = false
)