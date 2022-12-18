###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using MKL, LANSandBox, Plots, Flux, Distributions, Random, ProgressMeter
using ACTRModels
using Flux: params
using BSON: @save
include("functions.jl")
Random.seed!(285456)
###################################################################################################
#                                     Generate Training Data
###################################################################################################
# number of parameter vectors for training 
n_parms = 10_000
# number of sampled data points per parameter vector
n_samples = 100
# number of slot-values, which determines the response mapping
n_vals = (;n1=3,n2=3)
# model type: static no base level learning; dynamic: base level learning 
model_type = StaticModel
# fixed ACT-R parameters
fixed_parms = (τ = -10.0,s = 0.4, mmp=true,noise = true)
# generate training data 
sim_data = mapreduce(_ -> 
                    make_training_data(model_type; n_samples, n_vals..., fixed_parms...), 
                    hcat, 1:n_parms)
# input values
train_x = sim_data[1:6,:]
# log likelihoods
train_y = sim_data[end,:]'
train_data = Flux.Data.DataLoader((train_x, train_y), batchsize=1000)
###################################################################################################
#                                     Generate Training Data
###################################################################################################
# number of parameter vectors for training 
n_parms_test = 1000
# generate test data 
sim_data = mapreduce(_ -> 
                    make_training_data(model_type; n_samples, n_vals..., fixed_parms...), 
                    hcat, 1:n_parms_test)
# inputs
test_x = sim_data[1:6,:]
# log likelihoods
test_y = sim_data[end,:]'
test_data = (x=test_x, y=test_y)
###################################################################################################
#                                        Create Network
###################################################################################################
# 6 nodes in input layer, 3 hidden layers, 1 node for output layer
model = Chain(
    Dense(6, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 120, tanh),
    Dense(120, 1, identity)
)

# check our model
params(model)

# loss function
loss_fn(a, b) = Flux.huber_loss(model(a), b) 

# optimization algorithm 
opt = ADAM(0.001)
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
#@save "base_level_learning.bson" model
###################################################################################################
#                                      Plot Training
###################################################################################################
# plot the loss data
loss_plt = plot(1:n_epochs, train_loss, xlabel="Epochs", ylabel="Loss (huber)", label="training")
plot!(1:n_epochs, test_loss, label="test")

pred_y = model(test_x)
residual = pred_y .- test_y

scatter(
    pred_y', 
    test_y', 
    xlabel = "true density", 
    ylabel = "predicted density", 
    grid = false,
    leg = false
)

scatter(
    test_y', 
    residual', 
    xlabel = "true probability", 
    ylabel = "residual", 
    grid = false,
    leg = false
)