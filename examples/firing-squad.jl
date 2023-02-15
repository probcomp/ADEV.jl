flip = flip_mvd

# Basic causal model
function model()
    @adev begin
        let court = @sample(flip(0.5)),
            gunman_1 = false,
            gunman_2 = court,
            dead = gunman_1 || gunman_2

            dead ? 1 : 0
        end
    end
end

# Use existing tools for causal reasoning to derive 
# two interventions:

# Bribe gunman 1 not to shoot: do(gunman_1 = false)
function bribe_gunman()
    @adev begin
        let court = @sample(flip(0.5)),
            gunman_1 = false,
            gunman_2 = court,
            dead = gunman_1 || gunman_2

            dead ? 1 : 0
        end
    end
end

# Bribe court not to order firing: do(court = false)
function bribe_court()
    @adev begin
        let court = false,
            gunman_1 = court,
            gunman_2 = court,
            dead = gunman_1 || gunman_2

            dead ? 1 : 0
        end
    end
end


# Model for ADEV: theta parameterizes a policy
# for choosing who to bribe.
function firing_squad(theta)
    @adev begin
        if @sample(flip(sigmoid(theta)))
            @sample(bribe_court())
        else
            @sample(bribe_gunman())
        end
    end
end

sigmoid(x) = 1 / (1 + exp(-x))

# Optimize theta to minimize expected loss, using ADEV:
sgd(firing_squad; step_size=0.1, n_iters=1000, theta=0.5)

# Result: always choose to bribe the court!