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
n_parms = 40_000
# number of data points per parameter vector 
n_samples = 400
# training data
data = mapreduce(_ -> make_training_data(n_samples), hcat, 1:n_parms)
# true values 
labels = map(i -> pdf(Normal(data[1,i], data[2,i]), data[3,i]), 1:size(data,2))
labels = reshape(labels, 1, length(labels))
###################################################################################################
#                                      Plot Densities
###################################################################################################
# plot predictions against true values 
scatter(
    labels[:], 
    model(data)[:], 
    xlabel = "true density", 
    ylabel = "predicted density", 
    grid = false,
    leg = false
)

# interpolated densities 
plots = Plots.Plot[]
for i in 1:20
    parms = rand_parms()
    dist = LBA(;parms...)

    x = range(0.1, 3.0, length=100)
    y1 = map(x -> pdf(dist, 1, x), x)
    y2 = mapreduce(x -> model([vcat(parms...)...,1,x]), vcat, x)

    p1 = plot(x, y1, grid=false, label="true")
    plot!(x, y2, label="predicted")
    push!(plots, p1)
    display(p1)
    sleep(.3)
end