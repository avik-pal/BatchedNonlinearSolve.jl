@kwdef struct BatchedNewtonRaphson{AD, LS, TC <: NLSolveTerminationCondition} <:
              AbstractBatchedNonlinearSolveAlgorithm
    autodiff::AD = AD.ForwardDiffBackend()
    linsolve::LS = nothing
    termination_condition::TC = NLSolveTerminationCondition(NLSolveTerminationMode.RelSafeBest;
        abstol=nothing,
        reltol=nothing)
end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::BatchedNewtonRaphson;
    abstol=nothing,
    reltol=nothing,
    maxiters=100,
    kwargs...)
    @assert !isinplace(prob) "In-place algorithms are not yet supported!"

    u, f, reconstruct = _construct_batched_problem_structure(prob)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

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
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode)
            return build_solution(prob, alg, reconstruct(xₙ), reconstruct(fₙ); retcode)
        end

        xₙ₋₁ .= xₙ
    end

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        xₙ = storage.u
        fₙ = f(xₙ)
    end

    return build_solution(prob,
        alg,
        reconstruct(xₙ),
        reconstruct(fₙ);
        retcode=ReturnCode.MaxIters)
end
