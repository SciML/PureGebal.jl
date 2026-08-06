using PureGebal: GebalWorkspace, balance!
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "support", "testutils.jl"))

# The norms driving GEBAL's scaling decisions are accumulated in squared
# magnitudes, which overflows to Inf near the top of the range and flushes to
# zero for denormals. Every one of these families exists because a guard for one
# of those two failures was missing at some point; a randn-only test suite
# passes with all of them broken.
const FAMILIES = (
    ("denormal 1e-320", (rng, T, n) -> T.(randn(rng, n, n)) .* T(1.0e-320)),
    (
        "mixed denormal/normal",
        (rng, T, n) -> (
            M = T.(randn(rng, n, n));
            M[rand(rng, n, n) .< 0.5] .*= T(1.0e-320); M
        ),
    ),
    ("near overflow 1e300", (rng, T, n) -> T.(randn(rng, n, n)) .* T(1.0e300)),
    (
        "full range 1e-320..1e300",
        (rng, T, n) -> T.(randn(rng, n, n)) .* T.(exp10.(rand(rng, -320:300, n, n))),
    ),
    (
        "exact powers of two",
        (rng, T, n) -> T.(exp2.(rand(rng, -40:40, n, n)) .* rand(rng, (-1, 1), n, n)),
    ),
)

@testset "extreme-magnitude input stays finite and balanced" begin
    rng = Random.Xoshiro(31337)
    for (name, gen) in FAMILIES, T in (Float64, ComplexF64)
        @testset "$name / $T" begin
            worse = 0
            total = 0
            for _ in 1:150
                n = rand(rng, 2:12)
                A = gen(rng, T, n)
                all(isfinite, A) || continue
                total += 1
                B = copy(A)
                ws = GebalWorkspace(B)
                balance!(B, ws)
                @test all(isfinite, B)
                @test all(isfinite, ws.scale)
                # Compare the achieved balance against LAPACK on the *same*
                # matrix. An absolute bound is not the invariant here: GEBAL
                # deliberately declines to scale when doing so would overflow, so
                # tens of octaves of residual imbalance are correct on full-range
                # input -- LAPACK leaves exactly the same. What must never happen
                # is coming out measurably worse than the reference, which is how
                # a dropped overflow guard manifests (it skips the scaling loop
                # entirely and leaves hundreds of octaves).
                Bl, _, _, _ = lapack_balance(A)
                imbalance(B) > imbalance(Bl) + 1.0e-9 && (worse += 1)
            end
            @test total > 0
            @test worse == 0
        end
    end
end

@testset "column norms survive over- and underflow" begin
    # Directly exercise the recovery path: a column whose squared magnitudes all
    # overflow, and one whose squared magnitudes all flush to zero.
    for (label, v) in (
            ("near overflow", fill(1.0e300, 12)),
            ("denormal", fill(1.0e-320, 12)),
            ("huge and tiny", [1.0e200; fill(1.0e-200, 11)]),
        )
        n = length(v)
        A = zeros(n, n)
        A[:, 1] .= v
        A[1, :] .= v
        A[2:end, 2:end] .= 1.0
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        @test all(isfinite, B) || error("non-finite output for $label")
        @test all(isfinite, ws.scale)
    end
end
