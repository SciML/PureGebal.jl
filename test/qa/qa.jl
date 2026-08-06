using SciMLTesting, PureGebal, JET, Test, LinearAlgebra

# ExplicitImports only sees an extension module once its trigger weakdep is
# loaded (`Base.get_extension` returns `nothing` otherwise), so loading
# ForwardDiff here is what puts PureGebalForwardDiffExt under QA.
using ForwardDiff

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension module actually exists rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    @test Base.get_extension(PureGebal, :PureGebalForwardDiffExt) !== nothing
end

# ForwardDiff declares no name `public` and exports none of its dual-number
# interface, so the only spelling available for the types and accessors an AD
# extension must use is the non-public one.
const FORWARDDIFF_NONPUBLIC = (:Dual, :value)

run_qa(
    PureGebal;
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = FORWARDDIFF_NONPUBLIC),
        all_qualified_accesses_are_public = (; ignore = FORWARDDIFF_NONPUBLIC),
    ),
)
