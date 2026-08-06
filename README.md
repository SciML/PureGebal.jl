# PureGebal.jl

[![Build Status](https://github.com/SciML/PureGebal.jl/workflows/CI/badge.svg)](https://github.com/SciML/PureGebal.jl/actions?query=workflow%3ACI)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.sciml.ai/PureGebal/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.sciml.ai/PureGebal/dev/)
[![ColPrac: Contributor's Guide](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A pure-Julia, allocation-free matrix balancing routine equivalent to LAPACK's
`xGEBAL`, generic over the matrix element type.

Balancing improves the accuracy of a subsequent eigenvalue computation or matrix
exponential. It applies a diagonal similarity transform whose factors are exact
powers of two — introducing no rounding error — optionally preceded by a
permutation that isolates eigenvalues already readable from triangular corners of
the matrix.

## Why

`LinearAlgebra.LAPACK.gebal!` allocates its `scale` vector on every call, which
rules it out of an allocation-free inner loop, and it accepts only the four BLAS
element types. `PureGebal` takes a caller-owned workspace, works for any number
type, and is measurably faster.

```julia
using PureGebal: GebalWorkspace, balance!, unbalance!

A = [1.0 1.0e6; 1.0e-6 1.0]
B = copy(A)
ws = GebalWorkspace(B)

balance!(B, ws)                  # allocation-free
exp_A = unbalance!(exp(B), ws)   # == exp(A)
```

Nothing is exported; import what you use or qualify it.

## Performance

Wall time against `LinearAlgebra.LAPACK.gebal!`, x86-64 Linux, Julia 1.12.6.
Lower is better; the balanced output is bit-identical.

| | OpenBLAS 0.3.29 | MKL 2025 | PureGebal |
|---|---|---|---|
| `Float64` n=30 | 4.2 µs | — | **2.6 µs** |
| `Float64` n=500 | 0.76 ms | 1.50 ms | **0.50 ms** |
| `Float64` n=4000 | 260 ms | 230 ms | **112 ms** |
| `ComplexF64` n=30 | 6.5 µs | — | **3.4 µs** |
| `ComplexF64` n=500 | 1.41 ms | 2.67 ms | **0.74 ms** |
| `ComplexF64` n=4000 | 338 ms | 333 ms | **184 ms** |

Two things account for it: accumulating the row and column norms in squared
magnitudes rather than through a per-element `hypot`, and using `abs2` rather
than `abs` so complex element types pay one `sqrt` per column instead of one per
entry. Both need explicit over/underflow guards to stay correct, which the test
suite pins down.

## Element types

`Float32`, `Float64`, `BigFloat`, their `Complex` counterparts, and any other
number type supporting the usual arithmetic. Number types that *wrap* a value
work through two hooks, `primalvalue` and `primaltype`; `ForwardDiff.Dual` is
supported out of the box via a package extension.

Balancing chooses its permutation and scale factors from primal values only.
That is exact rather than approximate: the factors are piecewise constant in the
matrix entries, so they carry no derivative of their own.

## Relationship to LAPACK

For BLAS element types this is intended as a drop-in replacement for
`LinearAlgebra.LAPACK.gebal!(job, A)`, and the tests assert bit-agreement on
random input.

Exact agreement on *every* input is neither promised nor possible: GEBAL is a
heuristic and LAPACK builds disagree with each other — OpenBLAS 0.3.23 and 0.3.29
return different permutations for the same 3×3 matrix. Where implementations
differ, each result is a valid balancing and the eigenvalues are preserved
exactly, so callers should not depend on which permutation comes back.

## License

MIT. The algorithm is a translation of LAPACK's `xGEBAL`, whose modified-BSD
license is reproduced in [LICENSE](LICENSE).
