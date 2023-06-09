module BatchedNonlinearSolve

using AbstractDifferentiation,
    ChainRulesCore,
    DiffEqBase,
    FiniteDiff,
    ForwardDiff,
    LinearSolve,
    NNlib,
    SciMLBase,
    Zygote
import CommonSolve: init, solve, solve!
import SciMLBase: build_solution

const AD = AbstractDifferentiation
const CRC = ChainRulesCore
const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()
const ∅p = SciMLBase.NullParameters()

abstract type AbstractBatchedNonlinearSolveAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end

include("utils.jl")
include("raphson.jl")
include("dfsane.jl")
include("broyden.jl")

export BatchedNewtonRalphson

end
