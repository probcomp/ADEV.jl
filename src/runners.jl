
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

function differentiate(adev_prog, val)
    ForwardDiff.partials(simulate_dual(adev_prog, val))[1]
end

function sgd(adev_prog; theta=0.0, n_iters=1000, step_size=0.01, N = 100)
    for _ in 1:n_iters
        theta -= step_size * estimate_derivative(adev_prog, theta; N = N)
    end
    return theta
end

export estimate_derivative, estimator_variance, simulate, simulate_dual, differentiate, sgd