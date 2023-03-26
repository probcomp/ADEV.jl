using ADEV, ForwardDiff

flip = flip_mvd

function rejection(theta)
    @adev begin
        let b = @sample(flip(theta)),
            p = b ? 0.6 : 0.4,
            q = b ? theta : 1-theta,
            M = 0.6/theta + 0.4/(1-theta),
            accept_prob = p / (M * q)
            if @sample(flip(accept_prob))
                # Accepted immediately: time = 0
                0
            else
                # Rejected: time = 1 + time to accept
                1 + @sample(rejection(theta))
            end
        end
    end
end

sgd(rejection; theta=0.3, n_iters=1000, step_size=0.01, N = 10)