using PureGebal: GebalWorkspace, balance!, balance, unbalance!, unbalance_eigvecs!
using LinearAlgebra
using Random
using Test

@testset "unbalance! undoes the similarity on a matrix function" begin
    rng = Random.Xoshiro(808)
    for T in (Float64, ComplexF64), n in (2, 5, 12)
        # Wide dynamic range so balancing actually has work to do, then
        # normalised: `exp` overflows to NaN well before the entries reach the
        # magnitudes the balancer itself handles, and this testset is about the
        # round trip, not about the exponential's domain.
        A = randn(rng, T, n, n) .* T.(exp10.(rand(rng, -5:5, n, n)))
        all(isfinite, A) || continue
        A ./= opnorm(A, 1)
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        @test unbalance!(exp(B), ws) ≈ exp(A) rtol = 1.0e-10
    end
end

@testset "unbalance! round-trips the matrix itself" begin
    rng = Random.Xoshiro(809)
    for n in (3, 8, 20)
        A = randn(rng, n, n)
        A[rand(rng, n, n) .< 0.5] .= 0
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        # Powers of two and permutations are both exact, so this is bit-exact.
        @test unbalance!(B, ws) == A
    end
end

@testset "unbalance_eigvecs! matches LAPACK xGEBAK" begin
    rng = Random.Xoshiro(810)
    for n in (4, 9), side in (:right, :left)
        A = randn(rng, n, n) .* exp10.(rand(rng, -5:5, n, n))
        all(isfinite, A) || continue
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        V = randn(rng, n, n)
        mine = unbalance_eigvecs!(copy(V), ws; side)
        theirs = LinearAlgebra.LAPACK.gebak!(
            'B', side === :right ? 'R' : 'L', ws.ilo, ws.ihi, ws.scale, copy(V)
        )
        @test mine == theirs
    end
end

@testset "eigenvectors of the balanced matrix back-transform correctly" begin
    rng = Random.Xoshiro(811)
    n = 7
    A = randn(rng, n, n) .* exp10.(rand(rng, -4:4, n, n))
    B = copy(A)
    ws = GebalWorkspace(B)
    balance!(B, ws)
    F = eigen(B)
    V = unbalance_eigvecs!(Matrix(F.vectors), ws)
    for k in 1:n
        v = V[:, k]
        @test A * v ≈ F.values[k] * v rtol = 1.0e-6
    end
end

@testset "balance is the non-mutating form of balance!" begin
    A = [1.0 1.0e6; 1.0e-6 1.0]
    original = copy(A)
    B, ws = balance(A)
    @test A == original
    Bm = copy(A)
    wsm = GebalWorkspace(Bm)
    balance!(Bm, wsm)
    @test B == Bm
    @test ws.scale == wsm.scale
end
