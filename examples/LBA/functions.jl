function sample_mixture(ν, A, k, τ)
    dist = LBA(;ν, A, k, τ)
    return rand() ≤ .8 ? rand(dist) : (rand(1:length(ν)), rand(Uniform(.1, 3)))
end

function rand_parms()
    ν = rand(Uniform(0, 5), 2)
    A = rand(Uniform(.05, 2))
    k = rand(Uniform(.05, 1))
    τ = rand(Uniform(.1, .6))
    return (;ν,A,k,τ)
end

function make_training_data(n)
    output = fill(0.0, 7, n)    
    ν,A,k,τ = rand_parms()
    x = map(_ -> sample_mixture(ν, A, k, τ), 1:n)
    for (i,v) in enumerate(x)
        output[:,i] = [ν...,A,k,τ,v...]
    end
    return output
end

function train_model(model, n_epochs, loss_fn, all_data, opt; show_progress=true)
    meter = Progress(n_epochs; enabled=show_progress)
    loss = zeros(n_epochs)
    @showprogress for i in 1:n_epochs
        Flux.train!(loss_fn, params(model), all_data, opt)
        loss[i] = loss_fn(data, labels)
        current_loss = round(loss[i], digits=4)
        next!(meter; showvalues = [(:iter,i),(:loss,current_loss)])
    end
    return loss
end

function gen_label(data)
    pdf(LBA(;ν=data[1:2,i],A=data[3,i], k=data[4,i], τ=data[5,i]), Int(data[6]), data[7,i])
end