# ADEV.jl

Experimental port of the [ADEV](https://github.com/probcomp/adev) library to Julia. Not yet documented or stable, and currently has low coverage of Julia language features.

## Usage

### Defining probabilistic compuations

ADEV exposes primitives, like `normal_reparam` and `flip_mvd`, which are functions returning "probabilistic computations." Bigger probabilistic computations can be composed using the syntax `@adev begin ... end`:

```julia
function my_program(theta)
    @adev begin
        let x = @sample(normal_reparam(theta, 1)),
            b = @sample(flip_mvd(sigmoid(theta)))

            if b
                @sample(normal_reparam(x, 1))
            else
                # Recurse
                theta * @sample(my_program(theta - 1))
            end
        end
    end
end
```

Within an `@adev begin ... end` block, the following constructs are currently supported:

- `@sample(e)`, where `e` is a supported expression evaluating to a probabilistic computation;
- `let x1 = e1, ..., xn = en; e; end`, where `e, e1, ..., en` are supported expressions;
- `if e_cond1; e1; elseif e_cond2; e2; ...; else; en; end`, where `e_cond1, e1, ..., en` are supported expressions;
- `e(e1, ..., en)`, where `e, e1, ..., en` are supported expressions.
- other Julia expressions (e.g. `arr[idx]`), so long as they are pure (no mutation, no randomness).

The main limitations, then, are:
- No mutation (e.g. `x += 4`) or simple imperative assignment (e.g. `x = 4`). Use `let` instead.
- No loops (e.g. `while` or `for`). Use recursion instead.
- No compound expressions that are not function calls, but not pure (e.g. `my_array[@sample(uniform(1:10))]`). Use `let idx = @sample(uniform(1:10)); my_array[idx]; end` instead.
- No function calls with keyword arguments, or splatted arguments.

### Estimating derivatives

ADEV currently works with functions of type `Real -> ProbabilisticComputation{Real}`. For such a function, `simulate(f, x)` will simulate a value of `f(x)`, and `differentiate(f, x)` will estimate the derivative of $\mathbb{E}_{v \sim f(x)}[v]$ with respect to $x$. To average more than one sample (e.g. 100), use `estimate_derivative(f, x; N=100)`. A simple version of stochastic gradient descent is implemented by `sgd(f; theta=0, step_size=0.01, n_iters=1000, N=100)`, where `N` is the number of samples to use for estimating the derivative at each iteration.