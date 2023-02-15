
# FLIP

function flip_primal(p::Float64)
    function run(kont)
        if rand() < p
            return kont(true)
        else
            return kont(false)
        end
    end
end

flip_reinforce(p::Float64)=flip_primal(p)
flip_enum(p::Float64)=flip_primal(p)
flip_mvd(p::Float64)=flip_primal(p)

export flip_reinforce, flip_enum, flip_mvd

# MEASURE-VALUED DERIVATIVE for FLIP

function flip_mvd(p::ForwardDiff.Dual{T}) where {T}
    function run(kont)
        function (wants_grad, rng)
            if !wants_grad
                return kont(rand(rng) < p)(false, rng)
            else
                b = rand(rng) < p
                # Copy the rng object
                rng_copy = copy(rng)
                l1 = kont(b)(true, rng)
                l2 = kont(!b)(false, rng_copy)
                # Compute dual number to return
                value_estimate = ForwardDiff.value(l1)
                mvd_estimate = (value_estimate - ForwardDiff.value(l2)) * (b ? 1 : -1)
                grad_estimate = ForwardDiff.partials(l1) + ForwardDiff.partials(p) * mvd_estimate
                return ForwardDiff.Dual{T}(value_estimate, grad_estimate)
            end
        end
    end
end

# ENUMERATION-BASED DERIVATIVE for FLIP

function flip_enum(p::ForwardDiff.Dual{T}) where {T}
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                return kont(rand(rng) < p)(false, rng)
            else
                # Share randomness in both branches
                rng_copy = copy(rng)
                true_result  = kont(true)(true, rng)
                false_result = kont(false)(true, rng_copy)
                return (p * true_result + (1-p) * false_result)
            end
        end
    end
end

# REINFORCE DERIVATIVE ESTIMATOR for FLIP

function flip_reinforce(p::ForwardDiff.Dual{T}) where {T}
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                return kont(rand(rng) < p)(false, rng)
            else
                b = rand(rng) < p
                result = kont(b)(true, rng)
                lpdf = b ? log(p) : log(1 - p)
                return ForwardDiff.Dual{T}(ForwardDiff.value(result), ForwardDiff.partials(result) + ForwardDiff.partials(lpdf) * ForwardDiff.value(result))
            end
        end
    end
end