function sample_mixture(ν, A, k, τ)
    if rand() ≤ .8 
        c,rt = 0,Inf
        while rt > 20
            dist = LBA(;ν, A, k, τ)
            c,rt = rand(dist)
        end
        return c,rt 
    end 
    return (rand(1:length(ν)), rand(Uniform(.1, 3)))
end

function rand_parms()
    ν = rand(Uniform(0, 5), 2)
    A = rand(Uniform(.02, 2))
    k = rand(Uniform(.05, 1))
    τ = rand(Uniform(.1, .6))
    return (;ν,A,k,τ)
end

function make_training_data(n)
    output = zeros(Float32, 7, n)    
    ν,A,k,τ = rand_parms()
    x = map(_ -> sample_mixture(ν, A, k, τ), 1:n)
    for (i,v) in enumerate(x)
        output[:,i] = [ν...,A,k,τ,v...]
    end
    return output
end

function gen_label(data)
    pdf(LBA(;ν=data[1:2],A=data[3], k=data[4], τ=data[5]), Int(data[6]), data[7])
end