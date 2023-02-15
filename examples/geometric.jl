using ADEV, ForwardDiff

flip = flip_mvd

function geom(theta)
    @adev begin
        if @sample(flip(theta))
            0
        else
            1 + @sample(geom(theta))
        end
    end
end

exact_expected_value(theta) = 1 / theta
function exact_derivative(theta)
    ForwardDiff.partials(exact_expected_value(ForwardDiff.Dual(theta, 1)))[1]
end

estimate_derivative(geom, 0.4; N=10000)
exact_derivative(0.4)