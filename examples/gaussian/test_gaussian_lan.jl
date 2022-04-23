###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Plots, Distributions, Random, ProgressMeter
using Flux: params
using BSON: @load
include("functions.jl")
Random.seed!(2202152)
@load "mymodel.bson" model
###################################################################################################
#                                      Plot Densities
###################################################################################################
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