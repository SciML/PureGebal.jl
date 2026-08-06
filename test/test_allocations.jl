using PureGebal: GebalWorkspace, balance!, unbalance!, unbalance_eigvecs!
using AllocCheck: check_allocs
using LinearAlgebra
using Random
using Test

# The reason this package exists: a caller-owned workspace makes repeated
# balancing allocation-free, which LAPACK's own wrapper cannot be because it
# allocates the scale vector on every call.
@testset "balance! does not allocate" begin
    rng = Random.Xoshiro(91)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (4, 30, 120)
        A = randn(rng, T, n, n)
        ws = GebalWorkspace(A)
        run!(B, w) = (balance!(B, w); nothing)
        B = copy(A)
        run!(B, ws)                       # compile and warm
        B .= A
        @test (@allocated run!(B, ws)) == 0
        # check = false skips the finiteness scan; still allocation-free.
        runf!(B, w) = (balance!(B, w; check = false); nothing)
        B .= A
        runf!(B, ws)
        B .= A
        @test (@allocated runf!(B, ws)) == 0
    end
end

@testset "the undo paths do not allocate" begin
    rng = Random.Xoshiro(92)
    n = 40
    A = randn(rng, n, n)
    B = copy(A)
    ws = GebalWorkspace(B)
    balance!(B, ws)
    X = exp(B)
    un!(Y, w) = (unbalance!(Y, w); nothing)
    un!(copy(X), ws)
    Y = copy(X)
    @test (@allocated un!(Y, ws)) == 0

    V = randn(rng, n, n)
    ev!(W, w) = (unbalance_eigvecs!(W, w); nothing)
    ev!(copy(V), ws)
    W = copy(V)
    @test (@allocated ev!(W, ws)) == 0
end

@testset "AllocCheck static analysis" begin
    # `ignore_throw = true`: argument validation always throws by design, and
    # those paths are unreachable for valid input. Everything else must be
    # statically allocation-free.
    A = randn(Random.Xoshiro(93), 32, 32)
    ws = GebalWorkspace(A)
    balance!(copy(A), ws)
    warm(B, w) = balance!(B, w; check = false)
    allocs = check_allocs(warm, (Matrix{Float64}, typeof(ws)); ignore_throw = true)
    @test isempty(allocs)
end
