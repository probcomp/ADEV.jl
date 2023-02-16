using Test
using ADEV

@testset "Pure function in ADEV macro" begin
    function simple(theta)
        @adev begin
            2 + theta
        end
    end
    @test simulate(simple, 3) == 5
    @test differentiate(simple, 0.5) == 1.0
end