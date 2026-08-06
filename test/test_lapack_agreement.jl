using PureGebal: GebalWorkspace, balance!
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "support", "testutils.jl"))

# GEBAL is a heuristic: LAPACK implementations disagree with each other on which
# permutation to pick for matrices that fully decouple (OpenBLAS 0.3.23 and
# 0.3.29 differ on the same 3x3), so exact agreement is asserted as a high rate
# on random input rather than as an invariant on every input. Where they differ,
# both are valid balancings -- which the eigenvalue check below is what actually
# pins down.
@testset "bit-agreement with LAPACK xGEBAL" begin
    rng = Random.Xoshiro(2024)
    families = (
        ("dense random", (T, n) -> randn(rng, T, n, n)),
        (
            "wide dynamic range",
            (T, n) -> randn(rng, T, n, n) .* T.(exp10.(rand(rng, -8:8, n, n))),
        ),
        (
            "70% zeros",
            (T, n) -> (M = randn(rng, T, n, n); M[rand(rng, n, n) .< 0.7] .= 0; M),
        ),
        ("upper triangular", (T, n) -> triu(randn(rng, T, n, n))),
        (
            "block isolated",
            (T, n) -> (M = randn(rng, T, n, n); M[1, 2:end] .= 0; M[2:end, 1] .= 0; M),
        ),
    )
    for (name, gen) in families, T in (Float32, Float64, ComplexF32, ComplexF64)
        agree = 0
        total = 0
        for _ in 1:60
            n = rand(rng, 2:24)
            A = gen(T, n)
            all(isfinite, A) || continue
            total += 1
            B = copy(A)
            ws = GebalWorkspace(B)
            balance!(B, ws)
            Bl, ilo, ihi, scale = lapack_balance(A)
            agree += (B == Bl && ws.ilo == ilo && ws.ihi == ihi && ws.scale == scale)
        end
        @test total > 0
        # Loose enough for permutation ties, tight enough that a real divergence
        # in the scaling loop fails the test.
        @test agree >= 0.9 * total
    end
end

@testset "balancing preserves eigenvalues exactly" begin
    rng = Random.Xoshiro(77)
    for T in (Float64, ComplexF64), n in (4, 9, 17)
        A = randn(rng, T, n, n) .* T.(exp10.(rand(rng, -6:6, n, n)))
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        # The scale factors are powers of two, so this is exact in floating
        # point -- the balanced matrix has the same eigenvalues as the original.
        ref = sort(ComplexF64.(eigvals(big.(A))), by = x -> (real(x), imag(x)))
        got = sort(ComplexF64.(eigvals(big.(B))), by = x -> (real(x), imag(x)))
        @test got ≈ ref rtol = 1.0e-25
    end
end

@testset "balancing improves the row/column imbalance" begin
    rng = Random.Xoshiro(5150)
    for n in (6, 15, 30)
        D = Diagonal(exp10.(range(-6, 6, n)))
        A = D * randn(rng, n, n) * inv(D)
        all(isfinite, A) || continue
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        @test imbalance(B) < imbalance(A)
        # GEBAL's stopping criterion leaves at most about one octave of
        # imbalance. This is the regression guard for a scaling loop that
        # silently does nothing.
        @test imbalance(B) <= 2.0
    end
end
