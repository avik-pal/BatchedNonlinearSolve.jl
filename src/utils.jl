function _get_tolerance(η, tc_η, ::Type{T}) where {T}
    fallback_η = real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)
    return ifelse(η !== nothing, η, ifelse(tc_η !== nothing, tc_η, fallback_η))
end

function _construct_batched_problem_structure(prob)
    return _construct_batched_problem_structure(prob.u0, prob.f, prob.p)
end
function _construct_batched_problem_structure(u0::AbstractArray, f, p)
    reconstruct(u) = reshape(u, size(u0))
    f_modified(u) = reshape(f(reconstruct(u), p), :, size(u0, ndims(u0)))
    return reshape(u0, :, size(u0, ndims(u0))), f_modified, reconstruct
end
function _construct_batched_problem_structure(u0::AbstractMatrix, f, p)
    f_modified(u) = f(u, p)
    return u0, f_modified, identity
end
function _construct_batched_problem_structure(u0::AbstractVector, f, p)
    reconstruct(u) = vec(u)
    f_modified(u) = reshape(f(reconstruct(u), p), :, 1)
    return reshape(u0, :, 1), f_modified, reconstruct
end

@views function _init_𝓙(x::AbstractMatrix)
    𝓙 = zeromatrix(x[:, 1])
    if ismutable(x)
        𝓙[diagind(𝓙)] .= one(eltype(x))
    else
        𝓙 .+= I
    end
    return repeat(𝓙, 1, 1, size(x, 2))
end

_retcode_from_storage(::Nothing) = ReturnCode.Success
function _retcode_from_storage(storage::NLSolveSafeTerminationResultWithState)
    if storage.return_code[] == DiffEqBase.NLSolveSafeTerminationReturnCode.Success
        return ReturnCode.Success
    else
        return ReturnCode.Terminated
    end
end
