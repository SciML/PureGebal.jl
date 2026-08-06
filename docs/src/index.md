# PureGebal.jl

A pure-Julia, allocation-free matrix balancing routine equivalent to LAPACK's
`xGEBAL`, generic over the matrix element type.

Balancing improves the accuracy of an eigenvalue computation or a matrix
exponential. It applies a diagonal similarity transform whose factors are exact
powers of two — so the transform introduces no rounding error at all — optionally
preceded by a permutation that isolates eigenvalues already readable from
triangular corners of the matrix.

`LinearAlgebra.LAPACK.gebal!` allocates its `scale` vector on every call, which
rules it out of an allocation-free inner loop, and it only accepts the four BLAS
element types. `PureGebal` takes a caller-owned workspace instead, and works for
any number type — including `BigFloat`, `Float32`, and `ForwardDiff.Dual`.

## Installation

```julia
using Pkg
Pkg.add("PureGebal")
```

## Quick start

```julia
using PureGebal: GebalWorkspace, balance!, unbalance!

A = [1.0 1.0e6; 1.0e-6 1.0]
B = copy(A)
ws = GebalWorkspace(B)

balance!(B, ws)      # B is now balanced; ws records the transform
exp_A = unbalance!(exp(B), ws)   # == exp(A), computed more accurately
```

Nothing is exported. Import the names you use, or qualify them as
`PureGebal.balance!`.

## Reusing the workspace

The workspace is the point: build it once, balance many times, allocate nothing.

```julia
using PureGebal: GebalWorkspace, balance!

ws = GebalWorkspace(Float64, 100)
A = randn(100, 100)

for _ in 1:1000
    B = copy(A)
    balance!(B, ws)          # no allocations
end
```

A workspace is mutated by every call, so concurrent calls must not share one.
Sharing across tasks leaves the balanced matrix correct but corrupts the recorded
transform, which then silently invalidates any later
[`unbalance!`](@ref PureGebal.unbalance!). Give each task its own.

## Element types

Anything that supports the usual arithmetic works:

```julia
using PureGebal: GebalWorkspace, balance!

A = BigFloat[1 1000000; 1//1000000 1]
ws = GebalWorkspace(A)
balance!(A, ws)
```

Number types that *wrap* a value — dual numbers, for instance — work through two
small hooks, [`primalvalue`](@ref PureGebal.primalvalue) and
[`primaltype`](@ref PureGebal.primaltype). `ForwardDiff.Dual` is supported out of
the box through a package extension:

```julia
using ForwardDiff, PureGebal

function balanced_exp(A)
    B = copy(A)
    ws = PureGebal.GebalWorkspace(B)
    PureGebal.balance!(B, ws)
    return sum(PureGebal.unbalance!(exp(B), ws))
end

ForwardDiff.gradient(balanced_exp, [1.0 1.0e3; 1.0e-3 1.0])
```

Balancing picks its permutation and scale factors from primal values only. That
is not an approximation: the factors are exact powers of two and piecewise
constant in the matrix entries, so they carry no derivative of their own, and
applying them scales value and partials identically.

## Errors and status

Argument problems — a non-square matrix, a workspace of the wrong size, offset
indexing — are programming errors and always throw.

Non-finite input is different. Balancing is undefined for it, and LAPACK's
`xGEBAL` has no error code to report it; the stdlib wrapper scans the matrix and
throws before calling in. `PureGebal` does the same by default, and lets you opt
out:

```julia
using PureGebal: GebalWorkspace, balance!, GEBAL_NONFINITE
using LinearAlgebra: issuccess

A = [1.0 Inf; 0.0 1.0]
ws = GebalWorkspace(A)

balance!(A, ws; check = false)   # no throw
issuccess(ws)                    # false
ws.info == GEBAL_NONFINITE       # true, and A is untouched
```

`check = false` also skips the `O(n^2)` scan, though that is not why you would
reach for it — the scan is a cheap predicate next to the balancing itself, and
the two paths measure the same. Its purpose is to remove the throwing branch when
the caller wants a status code rather than an exception.

## Relationship to LAPACK

For the four BLAS element types, `PureGebal` is intended to be a drop-in
replacement for `LinearAlgebra.LAPACK.gebal!(job, A)`, and its test suite asserts
bit-agreement on random input.

Exact agreement is not guaranteed on *every* input, and cannot be: GEBAL is a
heuristic, and different LAPACK builds disagree with each other. OpenBLAS 0.3.23
and 0.3.29 return different permutations for the same 3×3 matrix. Where
implementations differ, each result is a valid balancing — the eigenvalues are
preserved exactly either way — so code should not depend on which permutation
comes back.

Matrices spanning the full floating-point range are a related case: GEBAL
deliberately declines to scale when doing so would overflow, so a large residual
imbalance there is correct behaviour, not a defect. `PureGebal` matches LAPACK's
residual on such input.
