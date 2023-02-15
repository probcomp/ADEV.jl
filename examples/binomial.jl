using ADEV

flip = flip_mvd

function binom(theta, n)
    @adev begin
        if n == 0
            0
        else 
            let m = @sample(binom(theta, n-1)),
                b = @sample(flip(theta))
                if b
                    m + 1
                else
                    m
                end
            end
        end
    end
end