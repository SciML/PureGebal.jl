using PureGebal, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

A100 = randn(rng, 100, 100)
A1000 = randn(rng, 1000, 1000)
# Ill-scaled matrix: large row/col norms variation
A_ill = randn(rng, 200, 200) .* (10.0 .^ rand(rng, -4:4, 200, 1))

# =============================================================================
# Matrix balancing (gebal / LAPACK-style)
# =============================================================================

SUITE["balance"] = BenchmarkGroup()

SUITE["balance"]["balance_100"] = @benchmarkable PureGebal.balance($(copy(A100)))
SUITE["balance"]["balance_1000"] = @benchmarkable PureGebal.balance(
    $(copy(A1000))
)
SUITE["balance"]["balance_illscaled"] = @benchmarkable PureGebal.balance(
    $(copy(A_ill))
)
SUITE["balance"]["balance!"] = @benchmarkable PureGebal.balance!(A, ws) setup = (
    A = copy($A_ill); ws = PureGebal.GebalWorkspace(Float64, 200)
)

# =============================================================================
# Workspace path
# =============================================================================

SUITE["workspace"] = BenchmarkGroup()

SUITE["workspace"]["GebalWorkspace"] = @benchmarkable PureGebal.GebalWorkspace(
    Float64, 200
)
