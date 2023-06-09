# BatchedNonlinearSolve.jl

Batched Nonlinear Solvers for Machine Learning Applications

> **Note**: This package is mostly a testbed for comparing batched nonlinear solvers.
> It is not intended for general use. Users should instead use `NonlinearSolve.jl` directly.

## Available Algorithms

* `BatchedNewtonRaphson`: Newton-Raphson method.
* `BatchedDFSane`: Derivative-Free Sane method. Good choice for large-scale problems.
* `BatchedBroyden`: Broyden method.

## Quickstart

```julia
using BatchedNonlinearSolve, SciMLBase

f(x, p) = reshape(f(reshape(x, :, size(x, ndims(x))), p), size(x))

function f(x::AbstractMatrix, p)
    residual = similar(x)
    residual[1, :] .= x[1, :] .^ 2 .+ 2 .* x[1, :] .- 1
    residual[2, :] .= x[2, :] .^ 2 .- 4 .* x[2, :] .+ 4
    return residual
end

function f(x::AbstractVector, p)
    residual = similar(x)
    residual[1] = x[1] .^ 2 .+ 2 .* x[1] .- 1
    residual[2] = x[2] .^ 2 .- 4 .* x[2] .+ 4
    return residual
end

prob = NonlinearProblem(f, randn(Float32, 2, 5), nothing)
sol = solve(prob, BatchedNewtonRaphson())
sol.resid

prob = NonlinearProblem(f, randn(Float32, 2), nothing)
sol = solve(prob, BatchedNewtonRaphson())
sol.resid

prob = NonlinearProblem(f, randn(Float32, 1, 1, 2, 32), nothing)
sol = solve(prob, BatchedNewtonRaphson())
sol.resid
```
