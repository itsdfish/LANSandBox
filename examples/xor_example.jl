###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
# %% Import packages
using Plots, Flux
using Flux: params

# %% The training data X and the output y
X = [0f0 0 1 1; 0 1 0 1]
y = [0f0 1 1 0]

# Our model has 2 input, 2 neurons in hidden layer and 1 output. The activation function is a sigmoid
xor_model = Chain(
    Dense(2, 2, sigmoid),
    Dense(2, 1, sigmoid)
)

# Check our model
params(xor_model)

# Check our inputs and output
params(xor_model)[1]
params(xor_model)[3]

# Implements a loss functions using mse
loss_fn(a, b) = Flux.mse(xor_model(a), b) 

# We will use ADAM optimizer
opt = ADAM(0.001)

# Number of Epochs to run
N = 10000
loss = zeros(N)

# Train the model
for i in 1:N
    Flux.train!(loss_fn, params(xor_model), [(X, y)], opt)
    loss[i] = loss_fn(X, y)
    # if i % 100 == 0
    #     println("loss $loss[i]")
    # end
end

# %% Print loss and predict our data
@show loss_fn(X, y)
@show xor_model(X)

# Plot the loss data
loss_plt = plot(1:N, loss, xlabel="Epochs", legend=:none, ylabel="Loss (mse)");
#display("image/png", loss_plt)