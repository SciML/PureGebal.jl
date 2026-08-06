using PureGebal: GebalWorkspace, balance!, balance
using LinearAlgebra
using GenericSchur: GenericSchur
using Random
using Test

# GenericSchur.balance! is an independent pure-Julia translation of xGEBAL, so it
# is the reference for the element types LAPACK cannot handle at all.
@testset "matches GenericSchur across element types" begin
    rng = Random.Xoshiro(5)
    n = 16
    base = randn(rng, n, n)
    base[rand(rng, n, n) .< 0.4] .= 0
    for T in (Float32, Float64, BigFloat, ComplexF32, ComplexF64, Complex{BigFloat})
        A = T.(base)
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        ref, bal = GenericSchur.balance!(T.(base))
        @test ws.ilo == bal.ilo
        @test ws.ihi == bal.ihi
        @test B == ref
        @test eltype(ws.scale) === real(float(T))
    end
end

@testset "workspace element type" begin
    @test eltype(GebalWorkspace(Float32, 3)) === Float32
    @test eltype(GebalWorkspace(ComplexF32, 3)) === Float32
    @test eltype(GebalWorkspace(Complex{BigFloat}, 3)) === BigFloat
    # Integer input has no float of its own; `balance` promotes.
    B, ws = balance([1 1000; 1 1])
    @test eltype(B) === Float64
    @test eltype(ws.scale) === Float64
end

@testset "degenerate sizes" begin
    for n in (0, 1)
        A = randn(Random.Xoshiro(1), n, n)
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        @test B == A
        @test ws.ilo == 1
        @test ws.ihi == n
        @test LinearAlgebra.issuccess(ws)
    end
end
