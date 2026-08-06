using Documenter, PureGebal, LinearAlgebra

include("pages.jl")

makedocs(
    sitename = "PureGebal.jl",
    authors = "Chris Rackauckas",
    modules = [PureGebal],
    clean = true, checkdocs = :exports, linkcheck = true,
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/PureGebal/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/PureGebal.jl.git";
    push_preview = true
)
