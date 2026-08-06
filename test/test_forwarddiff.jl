using PureGebal: GebalWorkspace, balance!, unbalance!
using ForwardDiff: ForwardDiff
using LinearAlgebra
using Random
using Test

# Balancing is a similarity transform by a diagonal matrix of exact powers of
# two. The factors are piecewise constant in the matrix entries, so they carry no
# derivative of their own; the correct behaviour under AD is to choose them from
# the primal values and then scale value and partials identically.
@testset "Dual element type balances on primal values" begin
    rng = Random.Xoshiro(11)
    n = 8
    A = randn(rng, n, n) .* exp10.(rand(rng, -5:5, n, n))
    Bp = copy(A)
    wsp = GebalWorkspace(Bp)
    balance!(Bp, wsp)

    seed = randn(rng, n, n)
    Ad = ForwardDiff.Dual{:tag}.(A, seed)
    Bd = copy(Ad)
    wsd = GebalWorkspace(Bd)
    balance!(Bd, wsd)

    @test eltype(wsd.scale) === Float64
    @test wsd.ilo == wsp.ilo
    @test wsd.ihi == wsp.ihi
    @test wsd.scale == wsp.scale
    # The primal part of the Dual result is exactly the Float64 result.
    @test ForwardDiff.value.(Bd) == Bp
end

@testset "derivatives through balance! are correct" begin
    rng = Random.Xoshiro(12)
    n = 6
    A = randn(rng, n, n) .* exp10.(rand(rng, -4:4, n, n))
    seed = randn(rng, n, n)

    # Where the chosen powers of two are locally constant -- almost everywhere --
    # balancing is the linear map `B = inv(D) * P' * A * P * D`, so the partials
    # must be that same map applied to the seed. `unbalance!` inverts it exactly
    # (powers of two and permutations are both exact in floating point), so
    # undoing the transform on the partials must return the seed bit-for-bit.
    #
    # This is asserted instead of a finite difference on purpose: the natural
    # scalar objectives here have magnitude ~1e8 with derivatives ~1e2, so a
    # central difference loses all but about four digits to cancellation. The
    # identity below is both exact and a stronger statement.
    Ad = ForwardDiff.Dual{:tag}.(A, seed)
    Bd = copy(Ad)
    wsd = GebalWorkspace(Bd)
    balance!(Bd, wsd)
    partials = ForwardDiff.partials.(Bd, 1)
    @test unbalance!(copy(partials), wsd) == seed

    # A seed of zeros must stay zero: the scale factors contribute no derivative
    # of their own.
    Az = ForwardDiff.Dual{:tag}.(A, zero(A))
    Bz = copy(Az)
    balance!(Bz, GebalWorkspace(Bz))
    @test all(iszero, ForwardDiff.partials.(Bz, 1))

    # Gradients survive the full round trip, which is smooth and well-scaled.
    function roundtrip_sum(x::AbstractMatrix)
        B = copy(x)
        ws = GebalWorkspace(B)
        balance!(B, ws)
        return sum(abs2, unbalance!(B, ws))
    end
    @test roundtrip_sum(A) ≈ sum(abs2, A)
    @test ForwardDiff.gradient(roundtrip_sum, A) ≈
        ForwardDiff.gradient(x -> sum(abs2, x), A)
end

@testset "nested Duals peel to the underlying float" begin
    A = [1.0 1.0e6; 1.0e-6 1.0]
    inner = ForwardDiff.Dual{:i}.(A, one.(A))
    nested = ForwardDiff.Dual{:o}.(inner, one.(inner))
    B = copy(nested)
    ws = GebalWorkspace(B)
    balance!(B, ws)
    @test eltype(ws.scale) === Float64

    Bp = copy(A)
    wsp = GebalWorkspace(Bp)
    balance!(Bp, wsp)
    @test ws.scale == wsp.scale
    @test ForwardDiff.value.(ForwardDiff.value.(B)) == Bp
end
