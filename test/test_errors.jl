using PureGebal: GebalWorkspace, balance!, unbalance!, unbalance_eigvecs!,
    GEBAL_SUCCESS, GEBAL_NONFINITE
using LinearAlgebra
using Test

@testset "non-finite input" begin
    for bad in (Inf, -Inf, NaN)
        A = [1.0 bad; 0.0 1.0]
        ws = GebalWorkspace(A)
        # Default: throw, matching LinearAlgebra's own gebal! wrapper, which
        # rejects non-finite input before calling LAPACK.
        @test_throws ArgumentError balance!(copy(A), ws)

        # check = false: report through `info` and leave the matrix untouched,
        # so the call has no throwing path on the data.
        B = copy(A)
        balance!(B, ws; check = false)
        @test ws.info == GEBAL_NONFINITE
        @test !LinearAlgebra.issuccess(ws)
        @test isequal(B, A)
    end
end

@testset "info is reset by a subsequent success" begin
    ws = GebalWorkspace(Float64, 2)
    balance!([1.0 Inf; 0.0 1.0], ws; check = false)
    @test ws.info == GEBAL_NONFINITE
    balance!([1.0 1.0e6; 1.0e-6 1.0], ws)
    @test ws.info == GEBAL_SUCCESS
    @test LinearAlgebra.issuccess(ws)
end

@testset "argument validation always throws" begin
    ws = GebalWorkspace(Float64, 3)
    # Non-square.
    @test_throws DimensionMismatch balance!(randn(3, 4), ws)
    # Workspace sized for a different matrix.
    @test_throws DimensionMismatch balance!(randn(2, 2), ws)
    # Validation is not disabled by check = false -- these are programming
    # errors, not data conditions, and the kernel indexes with @inbounds.
    @test_throws DimensionMismatch balance!(randn(2, 2), ws; check = false)
    # Offset/mismatched indexing.
    @test_throws ArgumentError GebalWorkspace(Float64, -1)
end

@testset "downstream calls reject an unsuccessful workspace" begin
    ws = GebalWorkspace(Float64, 2)
    balance!([1.0 Inf; 0.0 1.0], ws; check = false)
    @test_throws ArgumentError unbalance!(randn(2, 2), ws)
    @test_throws ArgumentError unbalance_eigvecs!(randn(2, 2), ws)
end

@testset "unbalance_eigvecs! side validation" begin
    A = [1.0 1.0e6; 1.0e-6 1.0]
    ws = GebalWorkspace(A)
    balance!(copy(A), ws)
    @test_throws ArgumentError unbalance_eigvecs!(randn(2, 2), ws; side = :sideways)
    @test_throws DimensionMismatch unbalance_eigvecs!(randn(3, 2), ws)
end
