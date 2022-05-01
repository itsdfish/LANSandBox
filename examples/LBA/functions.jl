function sample_mixture(ν, A, k, τ)
    if rand() ≤ .8 
        c,rt = 0,Inf
        while rt > 20
            dist = LBA(;ν, A, k, τ)
            c,rt = rand(dist)
        end
        return rt
    end 
    return rand(Uniform(.1, 2))
end

function rand_parms()
    ν = rand(Uniform(0, 5), 2)
    A = rand(Uniform(.02, 2))
    k = rand(Uniform(.05, 1))
    τ = rand(Uniform(.1, .6))
    return (;ν,A,k,τ)
end

function make_training_data(n)
    ν,A,k,τ = rand_parms()
    n_options = length(ν)
    output = zeros(Float32, 5 + n_options, n * n_options)    
    rts = map(_ -> sample_mixture(ν, A, k, τ), 1:n)
    i = 1
    for rt in rts, j in 1:n_options
        output[:,i] = [ν...,A,k,τ,j,rt]
        i += 1
    end
    return output
end

function gen_label(data)
    pdf(LBA(;ν=data[1:2],A=data[3], k=data[4], τ=data[5]), Int(data[6]), data[7])
end