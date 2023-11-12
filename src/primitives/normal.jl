
# NORMAL

function normal_primal(mu::Float64, sigma::Float64)
    function (kont)
        return kont(randn() * sigma + mu)
    end
end

normal_mvd(mu::Float64, sigma::Float64) = normal_primal(mu, sigma)
normal_reparam(mu::Float64, sigma::Float64) = normal_primal(mu, sigma)
normal_mvd_is(mu::Float64, sigma::Float64) = normal_primal(mu, sigma)
normal_reinforce(mu::Float64, sigma::Float64) = normal_primal(mu, sigma)

export normal_mvd, normal_reparam, normal_mvd_is, normal_reinforce

# NORMAL_REPARAM

function normal_reparam(mu::ForwardDiff.Dual{T}, sigma::ForwardDiff.Dual{T}) where {T}
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                mu_val = ForwardDiff.value(mu)
                sigma_val = ForwardDiff.value(sigma)    
                return kont(randn(rng) * sigma_val + mu_val)(false, rng)
            end

            kont(randn(rng) * sigma + mu)(true, rng)
        end
    end
end

function normal_reparam(mu::ReverseDiff.TrackedReal, sigma::ReverseDiff.TrackedReal)
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                mu_val = ReverseDiff.value(mu)
                sigma_val = ReverseDiff.value(sigma)    
                return kont(randn(rng) * sigma_val + mu_val)(false, rng)
            end

            kont(randn(rng) * sigma + mu)(true, rng)
        end
    end
end


# NORMAL_MVD

# Can't yet take derivatives w.r.t. the standard deviation
function normal_mvd(mu::ForwardDiff.Dual{T}, sigma::Float64) where {T}
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                return kont(randn(rng) * sigma + mu)(false, rng)
            end
            mu_val = ForwardDiff.value(mu)
            sigma_val = ForwardDiff.value(sigma)
            val = randn(rng) * sigma_val + mu_val
            wei = rand(rng, Weibull(2, sqrt(2)))
            rng1, rng2 = copy(rng), copy(rng)
            primal_result = kont(ForwardDiff.Dual{T}(val, 0))(true, rng)
            plus_result   = kont(mu_val + wei * sigma_val)(false, rng1)
            minus_result  = kont(mu_val - wei * sigma_val)(false, rng2)

            val_estimate = ForwardDiff.value(primal_result)
            mvd_estimate = (ForwardDiff.value(plus_result) - ForwardDiff.value(minus_result))/(sigma_val * sqrt(2*pi))
            grad_estimate = ForwardDiff.partials(primal_result) + ForwardDiff.partials(mu) * mvd_estimate
            return ForwardDiff.Dual{T}(val_estimate, grad_estimate)
        end
    end
end

function normal_mvd(mu::ReverseDiff.TrackedReal, sigma::Float64)
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                return kont(randn(rng) * sigma + mu)(false, rng)
            end
            mu_val = ReverseDiff.value(mu)
            sigma_val = sigma
            val = randn(rng) * sigma_val + mu_val
            wei = rand(rng, Weibull(2, sqrt(2)))
            rng1, rng2 = copy(rng), copy(rng)
            primal_result = kont(val)(true, rng)
            plus_result   = kont(mu_val + wei * sigma_val)(false, rng1)
            minus_result  = kont(mu_val - wei * sigma_val)(false, rng2)

            return phantom_gradient(mu, (plus_result - minus_result) / (sigma * sqrt(2*pi))) + primal_result
        end
    end
end


function normal_mvd_is(mu::ForwardDiff.Dual{T}, sigma::Float64) where {T}
    function (kont)
        function (wants_grad, rng)
            if !wants_grad
                return kont(randn(rng) * sigma + mu)(false, rng)
            end
            mu_val = ForwardDiff.value(mu)
            sigma_val = ForwardDiff.value(sigma)
            b = rand(rng) < 0.5
            wei = rand(rng, Weibull(2, sqrt(2)))
            rng1, rng2 = copy(rng), copy(rng)
            plus_result   = kont(mu_val + wei * sigma_val)(b, rng1)
            minus_result  = kont(mu_val - wei * sigma_val)(!b, rng2)

            proposal_density = logpdf(Weibull(2, sqrt(2)), wei) - log(sigma_val) - log(2)
            model_density = logpdf(Normal(mu_val, sigma_val), mu_val + wei * sigma_val * (b ? 1 : -1))
            weight = exp(model_density - proposal_density)

            val_estimate = b ? ForwardDiff.value(plus_result) : ForwardDiff.value(minus_result)
            val_estimate *= weight

            mvd_estimate = (ForwardDiff.value(plus_result) - ForwardDiff.value(minus_result))/(sigma_val * sqrt(2*pi))
            grad_estimate = ForwardDiff.partials(mu) * mvd_estimate + (b ? ForwardDiff.partials(plus_result) : ForwardDiff.partials(minus_result)) * weight

            return ForwardDiff.Dual{T}(val_estimate, grad_estimate)
        end
    end
end

# NORMAL_REINFORCE

function normal_reinforce(mu::Union{Float64,ForwardDiff.Dual{T}}, sigma::Union{Float64,ForwardDiff.Dual{T}}) where {T}
    function (kont)
        function (wants_grad, rng)
            mu_val = ForwardDiff.value(mu)
            sigma_val = ForwardDiff.value(sigma)
            if !wants_grad
                return kont(randn(rng) * sigma_val + mu_val)(false, rng)
            else
                x = randn(rng) * sigma_val + mu_val
                result = kont(ForwardDiff.Dual{T}(x,0))(true, rng)
                lpdf = logpdf(Normal(mu, sigma), x)
                return ForwardDiff.Dual{T}(ForwardDiff.value(result), ForwardDiff.partials(result) + ForwardDiff.partials(lpdf) * ForwardDiff.value(result))
            end
        end
    end
end


function normal_reinforce(mu::Union{Float64,ReverseDiff.TrackedReal}, sigma::Union{Float64,ReverseDiff.TrackedReal})
    function (kont)
        function (wants_grad, rng)
            mu_val = ReverseDiff.value(mu)
            sigma_val = ReverseDiff.value(sigma)
            if !wants_grad
                return kont(randn(rng) * sigma_val + mu_val)(false, rng)
            else
                x = randn(rng) * sigma_val + mu_val
                # TODO: do we need to ensure that the x we pass is 'tracked'?
                result = kont(x)(true, rng)
                lpdf = logpdf(Normal(mu, sigma), x)
                return phantom_gradient(lpdf, result) + result
            end
        end
    end
end