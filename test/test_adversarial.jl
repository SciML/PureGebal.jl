using PureGebal: GebalWorkspace, balance!, unbalance!
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
            lossy = 0
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
                # is coming out measurably worse without preserving the original
                # matrix exactly. PureGebal may decline a LAPACK scaling step
                # when that step would irreversibly round a subnormal component.
                Bl, _, _, _ = lapack_balance(A)
                if imbalance(B) > imbalance(Bl) + 1.0e-9
                    recovered = unbalance!(copy(B), ws)
                    recovered == A || (lossy += 1)
                end
            end
            @test total > 0
            @test lossy == 0
        end
    end
end

@testset "scaling preserves active-block subnormals" begin
    u = nextfloat(0.0)
    q = sqrt(u)
    a = 3q / 4
    b = 2q
    A = [0.0 a 0.0; b 0.0 u; u 0.0 0.0]
    B = copy(A)
    ws = GebalWorkspace(B)
    balance!(B, ws)

    @test !iszero(B[3, 1])
    @test !iszero(det(big.(B)))
    @test unbalance!(copy(B), ws) == A
end

@testset "unbalancing applies each net scale once" begin
    u = nextfloat(0.0)
    q = sqrt(u)
    a = 3q / 4
    b = 2q
    cases = (
        [0.0 a u 0.0; b 0.0 0.0 0.0; 0.0 0.0 0.0 a; 0.0 u b 0.0],
        [u a; b 0.0],
    )

    for A in cases
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        @test unbalance!(copy(B), ws) == A
    end
end

@testset "integer zeros and BigFloat boundary norms" begin
    for T in (Int, Complex{Int})
        A = zeros(T, 2, 2)
        ws = GebalWorkspace(A)
        @test balance!(A, ws) === ws
    end

    setprecision(BigFloat, 256) do
        q = sqrt(floatmin(BigFloat))
        A = BigFloat[0 3q / 8; 3q / 4 0]
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        @test ws.scale == ones(BigFloat, 2)
        @test B == A
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

@testset "subnormal norms and accumulated scales match LAPACK" begin
    cases = (
        [0.0 8.65e-163; 1.655e-162 0.0],
        ComplexF64[0 8.65e-163im; 1.655e-162 0],
        Float32[0 4.0e-9; 1.0e-37 0],
        [0.0 0.1; 1.5e-323 0.0],
        Matrix(Tridiagonal(fill(1.0e-20, 63), zeros(64), fill(1.0e20, 63))),
    )

    for A in cases
        B = copy(A)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        Bl, ilo, ihi, scale = lapack_balance(A)

        @test B == Bl
        @test ws.ilo == ilo
        @test ws.ihi == ihi
        @test ws.scale == scale
        @test all(x -> isfinite(x) && !iszero(x), ws.scale)
    end
end

@testset "scaling preserves isolated couplings and subnormal diagonals" begin
    # Row 1 is scaled inside the balancing block, but its coupling to the
    # isolated third index is also modified. The scale guard must see it and
    # decline a transformation that would overflow it.
    A = [
        0.0 1.0e-162 floatmax(Float64)
        2.65e-162 0.0 nextfloat(0.0)
        0.0 0.0 1.0
    ]
    B = copy(A)
    ws = GebalWorkspace(B)
    balance!(B, ws)
    @test B[1, 3] == A[1, 3]
    @test isfinite(B[1, 3])
    @test B[2, 3] == A[2, 3]

    recovered = copy(B)
    unbalance!(recovered, ws)
    @test recovered == A

    # A nonzero subnormal may survive the shrinking pass but still fail to
    # round-trip through the reciprocal expansion. Keep the odd component
    # intact instead of accepting a transformed value that cannot be undone.
    odd = nextfloat(0.0)
    E = [0.0 1.0e-162 floatmax(Float64); 2.65e-162 0.0 3odd; 0.0 0.0 1.0]
    F = copy(E)
    ws = GebalWorkspace(F)
    balance!(F, ws)
    recovered = copy(F)
    unbalance!(recovered, ws)
    @test recovered == E

    G = ComplexF64[0 1.0e-162 floatmax(Float64); 2.65e-162 0 2odd + 3odd * im; 0 0 1]
    H = copy(G)
    ws = GebalWorkspace(H)
    balance!(H, ws)
    recovered = copy(H)
    unbalance!(recovered, ws)
    @test recovered == G

    # Rounding can also occur exactly at the normal boundary: a value just
    # below 2floatmin rounds up to floatmin when it is halved. Exercise both
    # shrinking spans and precisions, while retaining an exact 4u control.
    for T in (Float32, Float64)
        u = nextfloat(zero(T))
        q = sqrt(u)
        a = T(0.45) * q
        b = T(1.2) * q
        for transpose_fixture in (false, true)
            cases = ((prevfloat(T(2) * floatmin(T)), false), (T(4) * u, true))
            for (x, should_scale) in cases
                J = T[0 a floatmax(T); b 0 x; 0 0 1]
                transpose_fixture && (J = copy(transpose(J)))
                K = copy(J)
                ws = GebalWorkspace(K)
                balance!(K, ws)
                recovered = copy(K)
                unbalance!(recovered, ws)
                @test recovered == J
                @test (ws.scale[2] != one(T)) == should_scale
            end
        end
    end

    # A diagonal entry receives reciprocal multipliers during a similarity
    # transform. Leaving it untouched prevents a subnormal value from flushing
    # to zero between the two exact operations.
    C = [nextfloat(0.0) 2.65e-162; 1.0e-162 0.0]
    D = copy(C)
    ws = GebalWorkspace(D)
    balance!(D, ws; permute = false)
    @test D[1, 1] == C[1, 1]
    @test tr(D) == tr(C)
end
