using LinearAlgebra: LinearAlgebra

# Reference balancing through the stdlib LAPACK wrapper. Only defined for the
# four BLAS element types; everything else is checked against GenericSchur or
# against structural invariants.
function lapack_balance(A::AbstractMatrix{T}) where {T}
    B = copy(A)
    ilo, ihi, scale = LinearAlgebra.LAPACK.gebal!('B', B)
    return B, Int(ilo), Int(ihi), scale
end

# How far the balanced matrix is from having equal row and column norms, in
# octaves, computed in `BigFloat` so the metric itself cannot over/underflow.
# GEBAL's whole purpose is to drive this down; a regression that silently skips
# the scaling loop shows up here and nowhere else.
function imbalance(A::AbstractMatrix)
    n = size(A, 1)
    worst = 0.0
    for i in 1:n
        c = sqrt(sum(j -> abs2(big(A[j, i])), 1:n))
        r = sqrt(sum(j -> abs2(big(A[i, j])), 1:n))
        (iszero(c) || iszero(r)) && continue
        worst = max(worst, abs(Float64(log2(c / r))))
    end
    return worst
end
