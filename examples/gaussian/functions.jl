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

function train_model(
    model, 
    n_epochs, 
    loss_fn, 
    train_data, 
    train_x, 
    train_y, 
    test_data, 
    opt; 
    show_progress = true
    )
    meter = Progress(n_epochs; enabled=show_progress)
    train_loss = zeros(n_epochs)
    test_loss = zeros(n_epochs)
    max_loss = -Inf
    min_loss = Inf
    @showprogress for i in 1:n_epochs
        Flux.train!(loss_fn, params(model), train_data, opt)
        train_loss[i] = loss_fn(train_x, train_y)
        test_loss[i] = loss_fn(test_data.x, test_data.y)
        loss = round(train_loss[i], digits=4)
        max_loss = loss >  max_loss ? loss : max_loss 
        min_loss = loss <  min_loss ? loss : min_loss
        values = [(:iter,i),(:loss,loss), (:max_loss, max_loss), (:min_loss,min_loss)]
        next!(meter; showvalues = values)
    end
    return train_loss,test_loss
end