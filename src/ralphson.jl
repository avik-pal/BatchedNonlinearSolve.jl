struct BatchedNewtonRalphson{AD, LS, TC <: NLSolveTerminationCondition} <:
       AbstractBatchedNonlinearSolveAlgorithm
    autodiff::AD
    linsolve::LS
    termination_condition::TC
end

function BatchedNewtonRalphson(;
    autodiff=AD.ForwardDiffBackend(),
    linsolve=nothing,
    termination_condition=NLSolveTerminationCondition(NLSolveTerminationMode.RelSafeBest;
        abstol=nothing,
        reltol=nothing))
    return BatchedNewtonRalphson{
        typeof(autodiff),
        typeof(linsolve),
        typeof(termination_condition),
    }(autodiff,
        linsolve,
        termination_condition)
end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::BatchedNewtonRalphson;
    abstol=nothing,
    reltol=nothing,
    maxiters=100,
    kwargs...)
    @assert !isinplace(prob) "In-place algorithms are not yet supported!"

    u, f, reconstruct = _construct_batched_problem_structure(prob)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ?
              NLSolveSafeTerminationResultWithState(; u) : nothing

    xₙ, xₙ₋₁, δx = copy(u), copy(u), copy(u)
    T = eltype(u)

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    termination_condition = tc(storage)

    for i in 1:maxiters
        fₙ, (𝓙,) = AD.value_and_jacobian(alg.autodiff, f, xₙ)

        iszero(fₙ) && return build_solution(prob,
            alg,
            reconstruct(xₙ),
            reconstruct(fₙ);
            retcode=ReturnCode.Success)

        solve(LinearProblem(𝓙, vec(fₙ); u0=vec(δx)), alg.linsolve; kwargs...)
        xₙ .-= δx

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            return build_solution(prob,
                alg,
                reconstruct(xₙ),
                reconstruct(fₙ);
                retcode=ReturnCode.Success)
        end

        xₙ₋₁ .= xₙ
    end

    return build_solution(prob,
        alg,
        reconstruct(xₙ),
        reconstruct(fₙ);
        retcode=ReturnCode.Maxiters)
end
