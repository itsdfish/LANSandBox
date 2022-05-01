function sample_mixture(μ, σ′)
    return rand() ≤ .8 ? rand(Normal(μ, σ′)) : rand(Uniform(-8, 8))
end

function rand_parms()
    μ = rand(Uniform(-3, 3))
    σ′ = rand(Uniform(.1, 2))
    return (;μ,σ′)
end

function make_training_data(n)
    output = zeros(Float32, 3, n)    
    μ,σ′ = rand_parms()
    x = map(_ -> sample_mixture(μ, σ′), 1:n)
    for (i,v) in enumerate(x)
        output[:,i] = [μ, σ′ ,v]
    end
    return output
end