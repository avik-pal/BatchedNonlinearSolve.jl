@kwdef struct BatchedDFSane{T, F, TC <: NLSolveTerminationCondition} <:
              AbstractBatchedNonlinearSolveAlgorithm
    σₘᵢₙ::T = 1.0f-10
    σₘₐₓ::T = 1.0f+10
    σ₁::T = 1.0f0
    M::Int = 10
    γ::T = 1.0f-4
    τₘᵢₙ::T = 0.1f0
    τₘₐₓ::T = 0.5f0
    nexp::Int = 2
    ηₛ::F = (f₁, k, x, F) -> f₁ ./ k .^ 2
    termination_condition::TC = NLSolveTerminationCondition(NLSolveTerminationMode.RelSafeBest;
        abstol=nothing,
        reltol=nothing)
    max_inner_iterations::Int = 1000
end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::BatchedDFSane,
    args...;
    abstol=nothing,
    reltol=nothing,
    maxiters=100,
    kwargs...)
    # tc = alg.termination_condition
    # mode = DiffEqBase.get_termination_mode(tc)

    # f = Base.Fix2(prob.f, prob.p)
    # x = float(prob.u0)

    # if batched
    #     batch_size = size(x, 2)
    # end

    # T = eltype(x)
    # σ_min = float(alg.σ_min)
    # σ_max = float(alg.σ_max)
    # σ_k = batched ? fill(float(alg.σ_1), 1, batch_size) : float(alg.σ_1)

    # M = alg.M
    # γ = float(alg.γ)
    # τ_min = float(alg.τ_min)
    # τ_max = float(alg.τ_max)
    # nexp = alg.nexp
    # η_strategy = alg.η_strategy

    # batched && @assert ndims(x)==2 "Batched SimpleDFSane only supports 2D arrays"

    # if SciMLBase.isinplace(prob)
    #     error("SimpleDFSane currently only supports out-of-place nonlinear problems")
    # end

    # atol = abstol !== nothing ? abstol :
    #        (tc.abstol !== nothing ? tc.abstol :
    #         real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5))
    # rtol = reltol !== nothing ? reltol :
    #        (tc.reltol !== nothing ? tc.reltol : eps(real(one(eltype(T))))^(4 // 5))

    # if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
    #     error("SimpleDFSane currently doesn't support SAFE_BEST termination modes")
    # end

    # storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
    #           nothing
    # termination_condition = tc(storage)

    # function ff(x)
    #     F = f(x)
    #     f_k = if batched
    #         sum(abs2, F; dims=1) .^ (nexp / 2)
    #     else
    #         norm(F)^nexp
    #     end
    #     return f_k, F
    # end

    # function generate_history(f_k, M)
    #     if batched
    #         history = similar(f_k, (M, length(f_k)))
    #         history .= reshape(f_k, 1, :)
    #         return history
    #     else
    #         return fill(f_k, M)
    #     end
    # end

    # f_k, F_k = ff(x)
    # α_1 = convert(T, 1.0)
    # f_1 = f_k
    # history_f_k = generate_history(f_k, M)

    # for k in 1:maxiters
    #     # Spectral parameter range check
    #     if batched
    #         @. σ_k = sign(σ_k) * clamp(abs(σ_k), σ_min, σ_max)
    #     else
    #         σ_k = sign(σ_k) * clamp(abs(σ_k), σ_min, σ_max)
    #     end

    #     # Line search direction
    #     d = -σ_k .* F_k

    #     η = η_strategy(f_1, k, x, F_k)
    #     f̄ = batched ? maximum(history_f_k; dims=1) : maximum(history_f_k)
    #     α_p = α_1
    #     α_m = α_1
    #     x_new = @. x + α_p * d

    #     f_new, F_new = ff(x_new)

    #     inner_iterations = 0
    #     while true
    #         inner_iterations += 1

    #         if batched
    #             # NOTE: This is simply a heuristic, ideally we check using `all` but that is
    #             #       typically very expensive for large problems
    #             norm(f_new) ≤ norm(@. f̄ + η - γ * α_p^2 * f_k) && break
    #         else
    #             f_new ≤ f̄ + η - γ * α_p^2 * f_k && break
    #         end

    #         α_tp = @. α_p^2 * f_k / (f_new + (2 * α_p - 1) * f_k)
    #         x_new = @. x - α_m * d
    #         f_new, F_new = ff(x_new)

    #         if batched
    #             # NOTE: This is simply a heuristic, ideally we check using `all` but that is
    #             #       typically very expensive for large problems
    #             norm(f_new) ≤ norm(@. f̄ + η - γ * α_p^2 * f_k) && break
    #         else
    #             f_new ≤ f̄ + η - γ * α_p^2 * f_k && break
    #         end

    #         α_tm = @. α_m^2 * f_k / (f_new + (2 * α_m - 1) * f_k)
    #         α_p = @. clamp(α_tp, τ_min * α_p, τ_max * α_p)
    #         α_m = @. clamp(α_tm, τ_min * α_m, τ_max * α_m)
    #         x_new = @. x + α_p * d
    #         f_new, F_new = ff(x_new)

    #         # NOTE: The original algorithm runs till either condition is satisfied, however,
    #         #       for most batched problems like neural networks we only care about
    #         #       approximate convergence
    #         batched && (inner_iterations ≥ alg.max_inner_iterations) && break
    #     end

    #     if termination_condition(F_new, x_new, x, atol, rtol)
    #         return SciMLBase.build_solution(prob,
    #             alg,
    #             x_new,
    #             F_new;
    #             retcode=ReturnCode.Success)
    #     end

    #     # Update spectral parameter
    #     s_k = @. x_new - x
    #     y_k = @. F_new - F_k

    #     if batched
    #         σ_k = sum(abs2, s_k; dims=1) ./ (sum(s_k .* y_k; dims=1) .+ T(1e-5))
    #     else
    #         σ_k = (s_k' * s_k) / (s_k' * y_k)
    #     end

    #     # Take step
    #     x = x_new
    #     F_k = F_new
    #     f_k = f_new

    #     # Store function value
    #     if batched
    #         history_f_k[k % M + 1, :] .= vec(f_new)
    #     else
    #         history_f_k[k % M + 1] = f_new
    #     end
    # end
    # return SciMLBase.build_solution(prob, alg, x, F_k; retcode=ReturnCode.MaxIters)
end
