###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using LANSandBox, MKL, Plots, Flux, Distributions, Random, ProgressMeter, SequentialSamplingModels
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
n_samples = 200
# training data
train_x = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
train_x = Float32.(train_x)
# true values 
train_y = map(i -> gen_label(train_x[:,i]), 1:size(train_x, 2))
train_y = Float32.(train_y)
train_y = reshape(train_y, 1, length(train_y))
train_data = Flux.Data.DataLoader((train_x, train_y), batchsize = 30_000)
###################################################################################################
#                                   Generate Test Data
###################################################################################################
n_parms_test = 1000
n_samples_test = 100
test_x = mapreduce(_ -> make_training_data(n_samples_test), hcat, 1:n_parms_test)
test_x = Float32.(test_x)
# true values 
test_y = map(i -> gen_label(test_x[:,i]), 1:size(test_x,2))
test_y = Float32.(test_y)
test_y = reshape(test_y, 1, length(test_y))
test_data = (x=test_x,y=test_y)
###################################################################################################
#                                        Create Network
###################################################################################################
# 7 nodes in input layer, 3 hidden layers, 1 node for output layer
model = Chain(
    Dense(7, 100, tanh),
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
@save "lba_model.bson" model
###################################################################################################
#                                      Plot Training
###################################################################################################
# plot the loss data
loss_plt = plot(1:n_epochs, train_loss, xlabel="Epochs", ylabel="Loss (huber)", label="training")
plot!(1:n_epochs, test_loss, label="test")

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