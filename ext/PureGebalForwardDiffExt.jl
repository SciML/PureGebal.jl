module PureGebalForwardDiffExt

using PureGebal: PureGebal
using ForwardDiff: ForwardDiff, Dual

# Balancing decisions (norm comparisons, zero tests, the isolation search) are
# made on primal values. The resulting scale factors are exact powers of two, so
# applying them to a `Dual` scales value and partials identically -- which is the
# correct derivative, since the factors are piecewise constant in the matrix
# entries and so contribute no derivative of their own. Peel nested `Dual`s all
# the way down to the underlying float.
PureGebal.primalvalue(d::Dual) = PureGebal.primalvalue(ForwardDiff.value(d))
PureGebal.primaltype(::Type{<:Dual{<:Any, V}}) where {V} = PureGebal.primaltype(V)

end
