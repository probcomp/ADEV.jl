using Test
using ADEV

@testset "Pure function in ADEV macro" begin
    function simple(theta)
        @adev begin
            2 + theta
        end
    end
    @test simple(3) == 2 + 3
end