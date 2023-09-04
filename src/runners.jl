
using StatsBase: mean, std

function estimate_derivative(adev_prog, val; N=100)
    mean(differentiate(adev_prog, val) for _ in 1:N)
end

function estimator_variance(adev_prog, val; N=100)
    std(differentiate(adev_prog, val) for _ in 1:N)
end

function simulate(adev_prog, val)
    simulate(adev_prog(val))
end

function simulate(adev_prog)
    adev_prog(x -> x)
end

function simulate_dual(adev_prog, val)
    adev_prog(ForwardDiff.Dual(val, 1))(x -> (wants_grad, rng) -> x)(true, Random.Xoshiro())
end

function simulate_reverse(adev_prog, args...)
    scalar_indices = [i for (i,a) in enumerate(args) if a isa Real]

    ReverseDiff.gradient((args...) -> adev_prog([i in scalar_indices ? a[1] : a for (i,a) in enumerate(args)]...)(r -> (wants_grad, rng) -> r)(true, Random.Xoshiro()), Tuple(i in scalar_indices ? [a] : a for (i,a) in enumerate(args)))
end

function differentiate(adev_prog, val)
    partials = ForwardDiff.partials(simulate_dual(adev_prog, val))
    if length(partials) == 0
        return 0.0
    else
        return partials[1]
    end
#    ForwardDiff.partials(simulate_dual(adev_prog, val))[1]
end


function gradient(adev_prog, vals...)
    return simulate_reverse(adev_prog, vals...)
end



function sgd(adev_prog; theta=0.0, n_iters=1000, step_size=0.01, N = 100)
    for _ in 1:n_iters
        theta -= step_size * estimate_derivative(adev_prog, theta; N = N)
    end
    return theta
end

export estimate_derivative, estimator_variance, simulate, simulate_dual, differentiate, sgd, simulate_reverse