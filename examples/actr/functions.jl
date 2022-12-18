abstract type AbstractModel end

mutable struct DynamicModel <: AbstractModel
    blc::Float64
    δ::Float64
    d::Float64
end

DynamicModel() = DynamicModel(0.0, 0.0, 0.0)

mutable struct StaticModel <: AbstractModel
    blc::Float64
    δ::Float64
end

StaticModel() = StaticModel(0.0, 0.0)

function rand_parms!(model::StaticModel)
    model.blc = rand(Uniform(0, 2))
    model.δ = rand(Uniform(0, 3))
    return nothing
end

function rand_parms!(model::DynamicModel)
    model.blc = rand(Uniform(0, 2))
    model.δ = rand(Uniform(0, 3))
    model.d = rand()
    return nothing
end

function make_training_data(model_type; n1, n2, n_samples, fixed_parms...)
    model = model_type()
    rand_parms!(model)
    return simulate(model; n_samples, n1, n2, fixed_parms...)
end

function populate_memory(;n1, n2)
    return [Chunk(;s1 = v1, s2 = v2) for v1 ∈ 1:n1 for v2 ∈ 1:n2]
end

function computeLL(model::StaticModel, choice, rt, cue1, cue2; n1, n2, fixed_parms...)
    (;blc,δ) = model
    # create a chunks
    chunks = populate_memory(;n1, n2)
    # add the chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, blc, δ, fixed_parms...)
    σ1 = actr.parms.s * pi / sqrt(3)
    compute_activation!(actr; s1 = cue1, s2=cue2)
    μ = -get_mean_activations(chunks)
    dist = LNR(;μ, σ=σ1, ϕ = 0.0)
    indices = find_indices(actr; s1=choice)
    log_probs = zeros(length(indices))
    for (i,idx) in enumerate(indices)
        log_probs[i] = logpdf(dist, idx, rt)
    end
    LL = Flux.logsumexp(log_probs)
end

function generate_data(model::StaticModel, cue1, cue2; n1, n2, fixed_parms...)
    (;blc,δ) = model
    # create a chunks
    chunks = populate_memory(;n1, n2)
    # add the chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, blc, δ, fixed_parms...)
    σ1 = actr.parms.s * pi / sqrt(3)
    compute_activation!(actr; s1 = cue1, s2=cue2)
    μ = -get_mean_activations(chunks)
    dist = LNR(;μ, σ=σ1, ϕ = 0.0)
    idx,rt = rand(dist)
    choice = chunks[idx].slots.s1
    return choice,rt
end

function simulate(model::StaticModel; n_samples, n1, n2, fixed_parms...)
    (;blc,δ) = model
    data = zeros(Float32, 7, n_samples)
    cue1,cue2 = rand(1:n1),rand(1:n2)
    for c ∈ 1:n_samples       
        choice,rt = generate_data(model, cue1, cue2; n1, n2, fixed_parms...)
        LL = computeLL(model, choice, rt, cue1, cue2; n1, n2, fixed_parms...)
        data[:,c] = [cue1 cue2 rt choice blc δ LL]
    end
    return data
end
