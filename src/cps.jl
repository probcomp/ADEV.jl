# Macro for defining probabilistic computations.
# Syntax is @adev begin ... end. The body of the 
# block may contain *functional* Julia code, as well 
# as invocations of other probabilistic computations
# using the @sample(computation) macro.

# Currently, the functional DSL supports "valid expressions," defined as those which:
# - are pure (no probabilistic effects or mutation)
# - are of the form @sample(e), where e is a valid expression that 
#   evaluates to an effectful computation
# - are of the form 
#     let x1 = e1, ..., xn = en
#        e
#     end, where e, e1, ..., en are valid expressions.
# - are of the form e(e1, ..., en), where e, e1, ..., en are valid expressions.
# - are of the form 
#     if e_cond1
#        e1
#     elseif e_cond2
#        e2
#     ...
#     else
#        en
#     end, where e_cond1, e1, ..., en are valid expressions.

# Main things that are disallowed:
# - Mutation (x += 4)
# - Loops (while, for) -- use recursion instead
# - Imperative assignment (x = 4) -- use `let` instead
# - Compound expressions that are not function calls, but not pure
#   (e.g. my_array[@sample(uniform(1:10))] -- use `let idx = @sample(uniform(1:10)); my_array[idx]; end` instead)
# - Function calls with keyword arguments, or splatted arguments.

macro adev(block)
    kont_name = gensym("kont")
    return esc(Expr(:->, kont_name, cps_transform_expr(block).expr(kont_name)))
end

# The macro relies on our continuation-passing transform of functional Julia code.

# cps_transform_expr
# Given a valid expression e, returns either:
#   - a NamedTuple of the form (is_pure = true, expr = e), if e is pure
#   - a NamedTuple of the form (is_pure = false, expr = kont -> e), if e is not pure.
#     `kont` is taken to be a Julia function of type Expr -> Expr.

function cps_transform_expr(e :: Symbol)
    return (is_pure = true, expr = e)
end

function cps_transform_expr(e :: Number)
    return (is_pure = true, expr = e)
end

function cps_transform_call(head, parameters, args)
    return (is_pure = false, expr = (kont) -> begin
            # TODO: handle splatted arguments.
            head = cps_transform_expr(head)
            head_name = gensym("head")
            args = [cps_transform_expr(arg) for arg in args]
            args_translated = [arg.is_pure ? arg.expr : gensym("arg") for arg in args]

            call_expression = Expr(:call, kont, Expr(:call, head.is_pure ? head.expr : head_name, args_translated...))
            # For each argument that is not pure, we need to prefix the expression
            for (arg, e) in zip(args, args_translated)
                if !arg.is_pure
                    call_expression = arg.expr(Expr(:->, e, call_expression))
                end
            end
            if !head.is_pure
                call_expression = head.expr(Expr(:->, head_name, call_expression))
            end
            return call_expression
    end)
end


# Returns a NamedTuple of the form (is_pure, expr)
function cps_transform_expr(e :: Expr)

    # Calls to Julia functions.
    # It is possible that either the callee or the arguments are the results of
    # effectful computation. If not, then we simply call the continuation on 
    # the result of the function call.
    if e.head == :call
        transformed = [cps_transform_expr(e) for e in e.args]
        if all(t.is_pure for t in transformed)
            return (is_pure = true, expr = e)
        else
            head = e.args[1]
            if length(e.args) > 1 && e.args[2] isa Expr && e.args[2].head == :parameters
                parameters = e.args[2]
                args = e.args[3:end]
            else
                parameters = nothing
                args = e.args[2:end]
            end
            return cps_transform_call(head, parameters, args)
        end
    end

    # Call to the @sample macro 
    if e.head == :macrocall
        if e.args[1] == Symbol("@sample")

            effectful_val = e.args[3]
            transformed = cps_transform_expr(effectful_val)
            if transformed.is_pure
                return (is_pure = false, expr = kont -> Expr(:call, effectful_val, kont))
            else
                effectful_val_name = gensym("effectful_val")
                return (is_pure = false, expr = kont_name -> transformed.expr(Expr(:->, effectful_val_name, Expr(:call, effectful_val_name, kont_name))))
            end
        end
    end

    # Assignment
    if e.head == Symbol("=")
        error("Cannot use imperative assignment in ADEV code. Use `let` instead.")
    end

    if e.head == :let
        assmt = e.args[1]
        body = e.args[2]

        if assmt.head == :block
            if length(assmt.args) == 0
                return cps_transform_expr(body)
            end
            if length(assmt.args) == 1
                assmt = assmt.args[1]
            else
                body = Expr(:let, Expr(:block, assmt.args[2:end]...), body)
                assmt = assmt.args[1]
            end
        end

        body = cps_transform_expr(body)

        # Just one assignment
        lhs = assmt.args[1]
        @assert lhs isa Symbol "`let` can only assign to variables, not $(lhs)"
        rhs = cps_transform_expr(assmt.args[2])
        if rhs.is_pure && body.is_pure
            return (is_pure = true, expr = e)
        elseif rhs.is_pure && !body.is_pure
            return (is_pure = false, expr = kont -> Expr(:let, assmt, body.expr(kont)))
        elseif !rhs.is_pure && body.is_pure
            return (is_pure = false, expr = kont -> rhs.expr(Expr(:->, lhs, Expr(:call, kont, body.expr))))
        elseif !rhs.is_pure && !body.is_pure
            return (is_pure = false, expr = kont -> rhs.expr(Expr(:->, lhs, body.expr(kont))))
        end
    end
    
    
    # Block of statements
    if e.head == :block
        stmts = [expr for expr in e.args if expr isa Expr]
        if length(stmts) == 1
            return cps_transform_expr(stmts[1])
        end
        if isempty(stmts)
            return (is_pure = true, expr = e)
        end
        println("Stmts: $stmts")
        transformed_first = cps_transform_expr(stmts[1])
        if transformed_first.is_pure
            transformed_rest = cps_transform_expr(Expr(:block, stmts[2:end]...))
            if transformed_rest.is_pure
                return (is_pure = true, expr = Expr(:block, transformed_first.expr, transformed_rest.expr))
            else
                return (is_pure = false, expr = kont -> Expr(:block, transformed_first.expr, transformed_rest.expr(kont)))
            end
        else
            transformed_rest = cps_transform_expr(Expr(:block, stmts[2:end]...))
            if transformed_rest.is_pure
                return (is_pure = false, expr = kont -> transformed_first.expr(Expr(:->, :_, Expr(:call, kont, transformed_rest.expr))))
            end
            return (is_pure = false, expr = kont -> transformed_first.expr(Expr(:->, :_, transformed_rest.expr(kont))))
        end
    end

    # If statements
    if e.head == :if || e.head == :elseif
        cond = e.args[1]
        true_branch = e.args[2]
        false_branch = (length(e.args) > 2) ? (e.args[3]) : nothing

        transformed_cond = cps_transform_expr(cond)
        transformed_true_branch = cps_transform_expr(true_branch)
        transformed_false_branch = isnothing(false_branch) ? nothing : cps_transform_expr(false_branch)

        if transformed_cond.is_pure && transformed_true_branch.is_pure && (isnothing(transformed_false_branch) || transformed_false_branch.is_pure)
            return (is_pure = true, expr = e)
        end

        cond_name = gensym("condition")
        true_expr = kont -> (transformed_true_branch.is_pure ? Expr(:call, kont, transformed_true_branch.expr) : transformed_true_branch.expr(kont))
        if isnothing(false_branch)
            if_args = kont -> [true_expr(kont)]
        else
            false_expr = kont -> (transformed_false_branch.is_pure ? Expr(:call, kont, transformed_false_branch.expr) : transformed_false_branch.expr(kont))
            if_args = kont -> [true_expr(kont), false_expr(kont)]
        end
        return (is_pure=false, expr = kont -> (transformed_cond.is_pure ? Expr(:if, transformed_cond.expr, if_args(kont)...) : transformed_cond.expr(Expr(:->, cond_name, Expr(:if, cond_name, if_args(kont)...)))))
        
    end

    @warn "Unknown statement type $(e.head), assuming pure."
    return (is_pure=true, expr=e)
end


export @adev
