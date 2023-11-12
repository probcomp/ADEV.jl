using ADEV
import Distributions
using StatsBase: mean

macro gen(block)
    transformed = ADEV.cps_transform_expr(block)
    kont_name = gensym("kont")
    expr = transformed.is_pure ? Expr(:call, kont_name, transformed.expr) : transformed.expr(kont_name)
    return esc(Expr(:->, :gentrace, Expr(:->, kont_name, expr)))
end

struct GenDist
    sim::Function
    est::Function
end

function make_gen_primitive(adev_primitive, density_function)
    function (args...)
        sim = @adev begin
            let x = @sample(adev_primitive(args...))
                (x, density_function(args..., x))
            end
        end
        est = x -> @adev begin
            density_function(args..., x)
        end
        GenDist(sim, est)
    end
end

flipReinforce = make_gen_primitive(ADEV.flip_reinforce, (p, b) -> b ? log(p) : log1p(-p))
flipMVD = make_gen_primitive(ADEV.flip_mvd, (p, b) -> b ? log(p) : log1p(-p))
flipEnum = make_gen_primitive(ADEV.flip_enum, (p, b) -> b ? log(p) : log1p(-p))
normalReparam = make_gen_primitive(ADEV.normal_reparam, (μ, σ, x) -> Distributions.logpdf(Distributions.Normal(μ, σ), x))
normalMVD = make_gen_primitive(ADEV.normal_mvd, (μ, σ, x) -> Distributions.logpdf(Distributions.Normal(μ, σ), x))
normalReinforce = make_gen_primitive(ADEV.normal_reinforce, (μ, σ, x) -> Distributions.logpdf(Distributions.Normal(μ, σ), x))

function simulate_trace(gen_dist, name)
    function (kont)
        @adev begin
            let (x, w1)        = @sample(gen_dist.sim),
                (y, trace, w2) = @sample(kont(x))
                (y, merge(trace, Dict(name => x)), w1 + w2)
            end
        end
    end
end

function make_density_tracer(observed_trace)
    function density_trace(gen_dist, name)
        function (kont)
            @adev begin
                let w1 = @sample(gen_dist.est(observed_trace[name])),
                    w2 = @sample(kont(observed_trace[name]))
                    w1 + w2
                end
            end
        end
    end
end

simulate(prog) = prog(simulate_trace)(x -> @adev begin (x, Dict(), 0.0) end)
density(prog, trace) = prog(make_density_tracer(trace))(x -> @adev begin 0.0 end)

function circleModel()
    @gen begin
        let x = @trace(normalReparam(0, 1), "x"),
            y = @trace(normalReparam(0, 1), "y"),
            z = @trace(normalReparam(sqrt(x^2 + y^2), 0.1), "z");
            z
        end
    end
end

function circleGuide(θ)
    @gen begin
        let x = @trace(normalReparam(θ[1], exp(θ[2])), "x"),
            y = @trace(normalReparam(θ[3], exp(θ[4])), "y")
            (x, y)
        end
    end
end

function elbo(θ)
    @adev begin 
        let (_, q_trace, logq) = @sample(simulate(circleGuide(θ))),
            trace = merge(q_trace, Dict("z" => 4)),
            logp = @sample(density(circleModel(), trace))
            logp - logq
        end
    end
end


# Gradient ascent on the ELBO
params = [0.,0.,0.,0.]
lr = 0.001
N = 1000
for i in 0:N
    params .+= ADEV.gradient(elbo, params)[1] .* lr

    if (i) % 100 == 0
        loss = mean(ADEV.simulate(elbo, params) for _ in 1:1000)
        println("Iter $(i), estimated loss = $loss")
    end
end

using Plots
begin
    scatter([(randn() * exp(params[2]) + params[1], randn() * exp(params[4]) + params[3]) for _ in 1:1000], xlims=(-5,5), ylims=(-5,5), legend=false, size=(500,500))
    plot!(x -> sqrt(16 - x^2), xlims=(-5,5), ylims=(-5,5), color=:red)
    plot!(x -> -sqrt(16 - x^2), xlims=(-5,5), ylims=(-5,5), color=:red)
end