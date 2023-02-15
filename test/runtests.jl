using Test
using ADEV

@testset "Pure function in ADEV macro" begin
    function simple()
        @adev begin
            2 + 3
        end
    end
    @test simple() == 2 + 3
end