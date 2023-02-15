using ADEV

flip = flip_reinforce

function figure_2_program(theta)
    @adev begin 
        if @sample(flip(theta))
            0
        else
            -theta / 2
        end
    end
end

function exact_expected_value(theta)
    (1 - theta) * (-theta / 2)
end

function exact_derivative(theta)
    ForwardDiff.partials(exact_expected_value(ForwardDiff.Dual(theta, 1)))[1]
end

sgd(figure_2_program; theta = rand(), n_iters = 1000, step_size = 0.1)