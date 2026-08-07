# API

```@meta
CurrentModule = PureGebal
```

```@docs
PureGebal
```

## Balancing

```@docs
PureGebal.balance!
PureGebal.balance
```

## Workspace

```@docs
PureGebal.GebalWorkspace
Base.eltype(::Type{PureGebal.GebalWorkspace{R}}) where {R}
LinearAlgebra.issuccess(::PureGebal.GebalWorkspace)
PureGebal.GEBAL_SUCCESS
PureGebal.GEBAL_NONFINITE
```

## Undoing the transform

```@docs
PureGebal.unbalance!
PureGebal.unbalance_eigvecs!
```

## Extending to wrapper number types

```@docs
PureGebal.primalvalue
PureGebal.primaltype
```
