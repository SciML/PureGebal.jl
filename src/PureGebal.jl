"""
    PureGebal

A pure-Julia, allocation-free matrix balancing routine equivalent to LAPACK's
`xGEBAL`, generic over the matrix element type.

Balancing applies an exact diagonal similarity transform — the scale factors are
powers of two, so the transform introduces no rounding error — optionally
preceded by a permutation that isolates eigenvalues already available from
triangular corners of the matrix. It improves the accuracy of a subsequent
eigenvalue computation or matrix exponential.

The entry points are [`balance!`](@ref) (in place, allocation-free given a
[`GebalWorkspace`](@ref)), [`balance`](@ref) (allocating convenience),
[`unbalance!`](@ref) (undo the similarity on a full matrix), and
[`unbalance_eigvecs!`](@ref) (the `xGEBAK` back-transform for eigenvectors).
"""
module PureGebal

# The balancing algorithm below is a translation of LAPACK's xGEBAL.
#   Copyright (c) 1992-2023 The University of Tennessee and The University of
#                           Tennessee Research Foundation. All rights reserved.
#   Copyright (c) 2000-2023 The University of California Berkeley.
#   Copyright (c) 2006-2023 The University of Colorado Denver.
# See LICENSE for the full modified-BSD terms this carries.

using LinearAlgebra: LinearAlgebra, checksquare

# `public` is Julia 1.11+; parse it dynamically so the package still loads on 1.10.
if VERSION >= v"1.11"
    eval(
        Meta.parse(
            "public GebalWorkspace, balance!, balance, unbalance!, unbalance_eigvecs!, " *
                "GEBAL_SUCCESS, GEBAL_NONFINITE, primalvalue, primaltype"
        )
    )
end

"""
    GEBAL_SUCCESS

Value of the `info` field of a [`GebalWorkspace`](@ref) after a successful
[`balance!`](@ref). Equal to `0`, following LAPACK's `info` convention.
"""
const GEBAL_SUCCESS = 0

"""
    GEBAL_NONFINITE

Value of the `info` field of a [`GebalWorkspace`](@ref) when [`balance!`](@ref)
was called with `check = false` on a matrix containing an `Inf` or `NaN`.

Balancing is undefined for non-finite input — LAPACK's `xGEBAL` has no error code
for it and its Julia wrapper rejects such input before the call — so the matrix
is left untouched and no balancing is performed. Positive `info` values are
Julia-side conditions; negative values would follow LAPACK's "illegal `i`-th
argument" convention, but this API validates arguments by throwing instead.
"""
const GEBAL_NONFINITE = 1

"""
    primalvalue(x) -> Number

Strip any derivative or uncertainty information from `x`, returning the plain
number underneath. The extension point through which a number type that wraps a
value — a dual number, say — becomes usable with [`balance!`](@ref).

Balancing chooses its permutation and its scale factors from primal values
alone. That is not an approximation: the factors are exact powers of two and
piecewise constant in the matrix entries, so they carry no derivative of their
own, and applying them scales a wrapped number's value and its derivative parts
identically.

Defined here for `Real` and `Complex`. Add a method for a wrapper type to make
it work with this package; `PureGebal` ships one for `ForwardDiff.Dual` as a
package extension. A method must return the wrapped value and must be free of
side effects. Implement [`primaltype`](@ref) alongside it.

# Returns

`x` with all wrapping removed, recursively.

# Examples

```julia
primalvalue(1.5)
primalvalue(1.0 + 2.0im)
```
"""
primalvalue(x::Real) = x
primalvalue(z::Complex) = complex(primalvalue(real(z)), primalvalue(imag(z)))

"""
    primaltype(::Type{T}) -> Type

The plain number type underlying `T`, the type-level counterpart of
[`primalvalue`](@ref). It determines the element type of a
[`GebalWorkspace`](@ref)'s `scale` vector, so balancing data is stored as plain
floating-point numbers rather than as wrapper values.

Add a method for a wrapper type alongside its [`primalvalue`](@ref) method.

# Returns

The unwrapped number type, recursively.

# Examples

```julia
primaltype(Float64)
primaltype(ComplexF32)
```
"""
primaltype(::Type{T}) where {T <: Real} = T
primaltype(::Type{Complex{T}}) where {T} = primaltype(T)

# Element type of the scale vector: the real floating-point type underlying `T`.
_scaletype(::Type{T}) where {T <: Number} = float(primaltype(T))

# Component max: |x| for real, max(|re|, |im|) for complex. Needed alongside the
# squared accumulators because `abs2` overflows to `Inf` and flushes to zero, in
# both cases destroying the very information required to recover; this involves
# no squaring and no `sqrt`, and is zero only when the entry is exactly zero.
@inline _cmax(x::Real) = abs(x)
@inline _cmax(z::Complex) = max(abs(real(z)), abs(imag(z)))

"""
    GebalWorkspace{R}

Caller-owned workspace for [`balance!`](@ref), holding every buffer the routine
needs so that balancing allocates nothing.

# Constructors

    GebalWorkspace(T::Type, n::Integer)
    GebalWorkspace(A::AbstractMatrix)

`T` is the matrix element type and `n` the matrix size; the second form takes
both from `A`. `R` is the real floating-point type underlying `T` (for a
`ForwardDiff.Dual` element type, the type its derivative information is built
on).

# Fields

  - `scale::Vector{R}`: the balancing data, in LAPACK's packed `xGEBAL` layout —
    entries `ilo:ihi` are the diagonal scale factors (exact powers of two) and
    the entries outside that range are permutation indices.
  - `ilo::Int`, `ihi::Int`: the balanced block is `A[ilo:ihi, ilo:ihi]`; rows and
    columns outside it were isolated by the permutation.
  - `info::Int`: status of the last [`balance!`](@ref); [`GEBAL_SUCCESS`](@ref)
    or [`GEBAL_NONFINITE`](@ref). See `LinearAlgebra.issuccess`.

# Returns

An uninitialized `GebalWorkspace` sized for `n`-by-`n` matrices with element
type `T`. Call [`balance!`](@ref) before reading its fields.

A workspace is mutated by every `balance!` call, so concurrent calls must each
use their own — see [`balance!`](@ref).

# Examples

```julia
A = [1.0 1.0e6; 1.0e-6 1.0]
ws = GebalWorkspace(A)
balance!(A, ws)
```
"""
mutable struct GebalWorkspace{R <: AbstractFloat}
    const scale::Vector{R}
    ilo::Int
    ihi::Int
    info::Int
end

function GebalWorkspace(::Type{T}, n::Integer) where {T <: Number}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative, got $n"))
    R = _scaletype(T)
    return GebalWorkspace{R}(Vector{R}(undef, n), 1, Int(n), GEBAL_SUCCESS)
end
GebalWorkspace(A::AbstractMatrix) = GebalWorkspace(eltype(A), checksquare(A))

"""
    eltype(::Type{GebalWorkspace{R}}) -> Type

The real floating-point type `R` the workspace stores its scale factors in — the
type underlying the matrix element type it was built for, so a workspace for a
`ComplexF32` or a `ForwardDiff.Dual` matrix has element type `Float32` or the
float its derivative information is built on.

# Examples

```julia
eltype(GebalWorkspace(ComplexF32, 4))
```
"""
Base.eltype(::Type{GebalWorkspace{R}}) where {R} = R

"""
    LinearAlgebra.issuccess(ws::GebalWorkspace) -> Bool

Whether the last [`balance!`](@ref) into `ws` succeeded, i.e. whether `ws.info`
is [`GEBAL_SUCCESS`](@ref).

# Examples

```julia
A = [1.0 1.0e6; 1.0e-6 1.0]
ws = GebalWorkspace(A)
balance!(A, ws)
LinearAlgebra.issuccess(ws)
```
"""
LinearAlgebra.issuccess(ws::GebalWorkspace) = ws.info == GEBAL_SUCCESS

# 2-norm from a squared accumulator. `cm` is the component max, which neither
# overflows nor underflows and so detects both failures of the squared
# accumulator; recovery rescales by an exact power of two, because `inv(cm)`
# would itself overflow when `cm` is denormal.
@inline function _norm_repair(A, ilo::Int, ihi::Int, i::Int, sq, cm, iscolumn::Bool)
    R = typeof(cm)
    nrm = sqrt(sq)
    if !isfinite(nrm) || (iszero(nrm) && !iszero(cm))
        e = exponent(cm)
        s = zero(R)
        @inbounds @simd for j in ilo:ihi
            a = iscolumn ? primalvalue(A[j, i]) : primalvalue(A[i, j])
            s += abs2(ldexp(real(a), -e)) + abs2(ldexp(imag(complex(a)), -e))
        end
        nrm = ldexp(sqrt(s), e)
    end
    return nrm
end

"""
    balance!(A, ws::GebalWorkspace; permute = true, scale = true, check = true) -> ws

Balance the square matrix `A` in place, writing the transform into `ws`.

Equivalent to LAPACK's `xGEBAL` with `job = 'B'` (or `'P'`/`'S'`/`'N'` for the
other combinations of `permute` and `scale`), but written in Julia, generic over
the element type, and allocation-free.

# Arguments

  - `A`: square matrix with one-based indexing and unit stride in its first
    dimension, overwritten with the balanced matrix.
  - `ws`: workspace from [`GebalWorkspace`](@ref), sized to match `A`.

# Keywords

  - `permute = true`: isolate eigenvalues by permuting rows and columns so the
    matrix acquires triangular corners.
  - `scale = true`: apply a diagonal similarity of exact powers of two making
    the row and column norms of the remaining block comparable.
  - `check = true`: scan `A` for non-finite entries first, throwing
    `ArgumentError` if any is found. Setting `check = false` skips the `O(n^2)`
    scan; a non-finite entry then leaves `A` untouched and sets `ws.info` to
    [`GEBAL_NONFINITE`](@ref) instead of throwing. Argument validation
    (dimensions, stride, indexing) always runs and always throws.

# Returns

The mutated `ws`, whose `scale`, `ilo`, `ihi`, and `info` fields describe the
transform applied to `A`. Undo it with [`unbalance!`](@ref) for a matrix, or
[`unbalance_eigvecs!`](@ref) for eigenvectors.

Because the scale factors are exact powers of two, balancing is an exact
similarity transform in floating point: it introduces no rounding error, and the
eigenvalues of the balanced matrix are those of the original.

`ws` is mutated, so concurrent `balance!` calls must not share one. Sharing a
workspace across tasks corrupts `scale` while leaving the balanced matrix itself
correct, which silently invalidates any later [`unbalance!`](@ref).

# Examples

```julia
A = [1.0 1.0e6; 1.0e-6 1.0]
ws = GebalWorkspace(A)
balance!(A, ws)
```
"""
function balance!(
        A::AbstractMatrix{T}, ws::GebalWorkspace{R};
        permute::Bool = true, scale::Bool = true, check::Bool = true
    ) where {T <: Number, R <: AbstractFloat}
    n = checksquare(A)
    Base.require_one_based_indexing(A)
    length(ws.scale) == n || throw(
        DimensionMismatch(
            "workspace holds $(length(ws.scale)) scale entries, matrix needs $n"
        )
    )
    n > 1 && stride(A, 1) == 1 || n <= 1 || throw(
        ArgumentError("matrix must have unit stride in its first dimension")
    )

    ws.ilo = 1
    ws.ihi = n
    ws.info = GEBAL_SUCCESS
    n == 0 && return ws

    finite = true
    @inbounds for a in A
        isfinite(primalvalue(a)) || (finite = false; break)
    end
    if !finite
        check && throw(ArgumentError("matrix contains Infs or NaNs"))
        ws.info = GEBAL_NONFINITE
        return ws
    end

    s = ws.scale
    fill!(s, one(R))
    ilo = 1
    ihi = n

    if permute
        ihi = _isolate_trailing!(A, s, n)
        if ihi > 1
            ilo = _isolate_leading!(A, s, n, ihi)
        end
    end
    if scale
        _scale_block!(A, s, n, ilo, ihi)
    end

    ws.ilo = ilo
    ws.ihi = ihi
    return ws
end

# Push rows with no off-diagonal entries to the bottom-right corner.
function _isolate_trailing!(A, s, n::Int)
    ihi = n + 1
    @inbounds while ihi > 1
        ihi -= 1
        exchange = false
        source = 0
        for j in ihi:-1:1
            exchange = true
            source = j
            for i in 1:ihi
                if i != j && !iszero(primalvalue(A[j, i]))
                    exchange = false
                    break
                end
            end
            exchange && break
        end
        exchange || break
        s[ihi] = source
        if source != ihi
            for i in 1:ihi
                A[i, source], A[i, ihi] = A[i, ihi], A[i, source]
            end
            for j in 1:n
                A[source, j], A[ihi, j] = A[ihi, j], A[source, j]
            end
        end
    end
    return ihi
end

# Push columns with no off-diagonal entries to the top-left corner.
function _isolate_leading!(A, s, n::Int, ihi::Int)
    ilo = 0
    @inbounds while ilo < n
        ilo += 1
        exchange = false
        source = 0
        for j in ilo:ihi
            exchange = true
            source = j
            for i in ilo:ihi
                if i != j && !iszero(primalvalue(A[i, j]))
                    exchange = false
                    break
                end
            end
            exchange && break
        end
        exchange || break
        s[ilo] = source
        if source != ilo
            for i in 1:ihi
                A[i, source], A[i, ilo] = A[i, ilo], A[i, source]
            end
            for j in ilo:n
                A[source, j], A[ilo, j] = A[ilo, j], A[source, j]
            end
        end
    end
    return ilo
end

# Iteratively rescale rows/columns by powers of two until row and column norms
# are within `factor` of balanced. Mirrors LAPACK's xGEBAL scaling loop.
function _scale_block!(A, s::Vector{R}, n::Int, ilo::Int, ihi::Int) where {R}
    radix = R(2)
    factor = R(0.95)
    sfmin1 = floatmin(R) / eps(R)
    sfmin2 = sfmin1 * radix
    sfmax2 = inv(sfmin2)
    converged = false
    while !converged
        converged = true
        @inbounds for i in ilo:ihi
            column_sq = zero(R)
            row_sq = zero(R)
            column_max_sq = zero(R)
            row_max_sq = zero(R)
            column_cmax = zero(R)
            row_cmax = zero(R)
            # One pass over squared magnitudes. `abs2` is branch-free and, for a
            # complex element type, avoids the per-element `sqrt` that `abs` and
            # `hypot` pay; the component max rides along to make the over/underflow
            # recovery below possible.
            @simd for j in ilo:ihi
                cj = primalvalue(A[j, i])
                rj = primalvalue(A[i, j])
                column_sq += abs2(cj)
                row_sq += abs2(rj)
                column_max_sq = max(column_max_sq, abs2(cj))
                row_max_sq = max(row_max_sq, abs2(rj))
                column_cmax = max(column_cmax, _cmax(cj))
                row_cmax = max(row_cmax, _cmax(rj))
            end
            column_norm = _norm_repair(A, ilo, ihi, i, column_sq, column_cmax, true)
            row_norm = _norm_repair(A, ilo, ihi, i, row_sq, row_cmax, false)
            column_max = sqrt(column_max_sq)
            row_max = sqrt(row_max_sq)
            # A max left at `Inf` silently trips the `sfmax2` guard below and
            # abandons the scaling loop, balancing nothing at all.
            if !isfinite(column_max) || (iszero(column_max) && !iszero(column_cmax))
                column_max = zero(R)
                for j in ilo:ihi
                    column_max = max(column_max, abs(primalvalue(A[j, i])))
                end
            end
            if !isfinite(row_max) || (iszero(row_max) && !iszero(row_cmax))
                row_max = zero(R)
                for j in ilo:ihi
                    row_max = max(row_max, abs(primalvalue(A[i, j])))
                end
            end

            (iszero(column_norm) || iszero(row_norm)) && continue

            g = row_norm / radix
            original_sum = column_norm + row_norm
            f = one(R)
            while column_norm < row_norm / radix
                if column_norm >= g || max(f, column_norm, column_max) >= sfmax2 ||
                        min(row_norm, g, row_max) <= sfmin2
                    break
                end
                f *= radix
                column_norm *= radix
                column_max *= radix
                row_norm /= radix
                g /= radix
                row_max /= radix
            end
            g = column_norm / radix
            while row_norm <= column_norm / radix
                if g < row_norm || max(row_norm, row_max) >= sfmax2 ||
                        min(f, column_norm, g, column_max) <= sfmin2
                    break
                end
                f /= radix
                column_norm /= radix
                g /= radix
                column_max /= radix
                row_norm *= radix
                row_max *= radix
            end
            column_norm + row_norm >= factor * original_sum && continue

            converged = false
            s[i] *= f
            invf = inv(f)
            for j in ilo:n
                A[i, j] *= invf
            end
            for j in 1:ihi
                A[j, i] *= f
            end
        end
    end
    return nothing
end

"""
    balance(A; permute = true, scale = true, check = true) -> (B, ws)

Allocating convenience wrapper for [`balance!`](@ref): balance a copy of `A`.

# Arguments

  - `A`: square matrix, not modified.

# Keywords

Identical to [`balance!`](@ref).

# Returns

`(B, ws)` where `B` is the balanced copy of `A` and `ws` the
[`GebalWorkspace`](@ref) describing the transform.

# Examples

```julia
B, ws = balance([1.0 1.0e6; 1.0e-6 1.0])
```
"""
function balance(A::AbstractMatrix; kwargs...)
    B = copyto!(similar(A, float(eltype(A))), A)
    ws = GebalWorkspace(eltype(B), checksquare(B))
    balance!(B, ws; kwargs...)
    return B, ws
end

"""
    unbalance!(X, ws::GebalWorkspace) -> X

Undo, in place, the similarity transform recorded in `ws` on the square matrix
`X`.

Use this on a matrix function of a balanced matrix: if `balance!` turned `A` into
`B`, then `unbalance!(f(B), ws)` is `f(A)`. For eigenvectors, use
[`unbalance_eigvecs!`](@ref) instead.

# Arguments

  - `X`: square matrix of the same size as the balanced matrix, overwritten.
  - `ws`: workspace returned by a successful [`balance!`](@ref).

# Returns

The mutated `X`.

# Examples

```julia
A = [1.0 1.0e6; 1.0e-6 1.0]
B = copy(A)
ws = GebalWorkspace(B)
balance!(B, ws)
unbalance!(exp(B), ws)   # == exp(A)
```
"""
function unbalance!(X::AbstractMatrix, ws::GebalWorkspace)
    n = checksquare(X)
    Base.require_one_based_indexing(X)
    length(ws.scale) == n || throw(
        DimensionMismatch("workspace is sized for $(length(ws.scale)), got $n")
    )
    LinearAlgebra.issuccess(ws) ||
        throw(ArgumentError("workspace does not hold a successful balancing"))
    s = ws.scale
    ilo, ihi = ws.ilo, ws.ihi
    @inbounds for j in ilo:ihi
        scj = s[j]
        for i in 1:n
            X[j, i] *= scj
        end
        for i in 1:n
            X[i, j] /= scj
        end
    end
    # The permutations are packed into `scale` outside `ilo:ihi`, and undoing
    # them means replaying the two isolation sweeps in reverse.
    @inbounds if ilo > 1
        for j in (ilo - 1):-1:1
            _rcswap!(j, Int(s[j]), X)
        end
    end
    @inbounds if ihi < n
        for j in (ihi + 1):n
            _rcswap!(j, Int(s[j]), X)
        end
    end
    return X
end

function _rcswap!(i::Integer, j::Integer, A::AbstractMatrix)
    i == j && return A
    @inbounds for k in axes(A, 2)
        A[i, k], A[j, k] = A[j, k], A[i, k]
    end
    @inbounds for k in axes(A, 1)
        A[k, i], A[k, j] = A[k, j], A[k, i]
    end
    return A
end

"""
    unbalance_eigvecs!(V, ws::GebalWorkspace; side = :right) -> V

Back-transform eigenvectors of a balanced matrix to eigenvectors of the original,
in place. The equivalent of LAPACK's `xGEBAK`.

# Arguments

  - `V`: matrix whose columns are eigenvectors of the balanced matrix,
    overwritten with eigenvectors of the original. It must have as many rows as
    the balanced matrix.
  - `ws`: workspace returned by a successful [`balance!`](@ref).

# Keywords

  - `side = :right`: `:right` for right eigenvectors, `:left` for left
    eigenvectors, which scale by the reciprocal factors.

# Returns

The mutated `V`.

# Examples

```julia
A = [1.0 1.0e6; 1.0e-6 1.0]
B = copy(A)
ws = GebalWorkspace(B)
balance!(B, ws)
V = eigen(B).vectors
unbalance_eigvecs!(V, ws)
```
"""
function unbalance_eigvecs!(V::AbstractMatrix, ws::GebalWorkspace; side::Symbol = :right)
    side in (:right, :left) ||
        throw(ArgumentError("side must be :right or :left, got $(repr(side))"))
    Base.require_one_based_indexing(V)
    n = length(ws.scale)
    size(V, 1) == n || throw(
        DimensionMismatch("eigenvector matrix has $(size(V, 1)) rows, expected $n")
    )
    LinearAlgebra.issuccess(ws) ||
        throw(ArgumentError("workspace does not hold a successful balancing"))
    s = ws.scale
    ilo, ihi = ws.ilo, ws.ihi
    m = size(V, 2)
    @inbounds for i in ilo:ihi
        f = side === :right ? s[i] : inv(s[i])
        for j in 1:m
            V[i, j] *= f
        end
    end
    @inbounds if ilo > 1
        for i in (ilo - 1):-1:1
            _swaprows!(V, i, Int(s[i]), m)
        end
    end
    @inbounds if ihi < n
        for i in (ihi + 1):n
            _swaprows!(V, i, Int(s[i]), m)
        end
    end
    return V
end

function _swaprows!(V::AbstractMatrix, i::Integer, k::Integer, m::Integer)
    i == k && return V
    @inbounds for j in 1:m
        V[i, j], V[k, j] = V[k, j], V[i, j]
    end
    return V
end

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        A = [1.0 1.0e6; 1.0e-6 1.0]
        B, ws = balance(A)
        unbalance!(copy(B), ws)
        V = [1.0 0.0; 0.0 1.0]
        unbalance_eigvecs!(V, ws)
        balance!(copy(A), GebalWorkspace(A); permute = false, scale = false)
    end
end

end # module
