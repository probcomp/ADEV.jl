
using ADEV

normal = ADEV.normal_mvd

function indicator(theta)
    @adev begin
        let x = @sample(normal(theta, 1.0))
            if x > 3
                (theta - 5)
            else
                0
            end
        end
    end
end

function exact_expected_value(theta)
    (1 - cdf(Normal(theta, 1), 3)) * (theta - 5)
end

function exact_derivative(theta)
    ForwardDiff.partials(exact_expected_value(ForwardDiff.Dual(theta, 1)))[1]
end

sgd(indicator; theta = 0.0, n_iters = 1000, step_size = 0.1)

thetas = collect(-5:0.1:8)
# MVD
normal = ADEV.normal_mvd
vars_mvd = [estimator_variance(indicator, theta; N=100000) for theta in thetas]
# REINFORCE
normal = ADEV.normal_reinforce
vars_reinforce = [estimator_variance(indicator, theta; N=100000) for theta in thetas]
# MVD IS
normal = ADEV.normal_mvd_is
vars_is = [estimator_variance(indicator, theta; N=100000) for theta in thetas]

using Plots
plot(thetas, vars_mvd, label="MVD")
plot!(thetas, vars_reinforce, label="REINFORCE")
plot!(thetas, vars_is, label="MVD IS")
plot!(thetas, exact_expected_value.(thetas), label="Exact")
plot!(thetas, exact_derivative.(thetas), label="Exact Derivative")

estimate_derivative(indicator, 6.0; N=1)