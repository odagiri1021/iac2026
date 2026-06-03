module LocalPoincareMap

using LinearAlgebra
using Statistics
using Serialization

export load_model, predict_next, iterate_map

function load_model(path::AbstractString)
    return deserialize(path)
end

function standardize_vector(x, μ, σ)
    xn = similar(x)
    for j in 1:length(x)
        xn[j] = (x[j] - μ[j]) / σ[j]
    end
    return xn
end

function destandardize_vector(xn, μ, σ)
    x = similar(xn)
    for j in 1:length(xn)
        x[j] = xn[j] * σ[j] + μ[j]
    end
    return x
end

function local_linear_predict(x_query_n, Xn, Yn, k_neighbors, ridge, bandwidth_factor)
    N = size(Xn, 1)
    nx = size(Xn, 2)
    ny = size(Yn, 2)

    k = min(k_neighbors, N)

    d2 = zeros(N)

    for i in 1:N
        s = 0.0
        for j in 1:nx
            dx = Xn[i, j] - x_query_n[j]
            s += dx * dx
        end
        d2[i] = s
    end

    idx = partialsortperm(d2, 1:k)
    d2k = d2[idx]

    d_med = median(sqrt.(d2k .+ eps()))
    h = bandwidth_factor * max(d_med, 1.0e-12)

    A = zeros(k, nx + 1)
    B = zeros(k, ny)
    w = zeros(k)

    for a in 1:k
        i = idx[a]

        A[a, 1] = 1.0

        for j in 1:nx
            A[a, j+1] = Xn[i, j] - x_query_n[j]
        end

        for j in 1:ny
            B[a, j] = Yn[i, j]
        end

        w[a] = exp(-d2[i] / (2.0 * h^2))
    end

    sqrtw = sqrt.(w)

    for a in 1:k
        A[a, :] .*= sqrtw[a]
        B[a, :] .*= sqrtw[a]
    end

    R = A' * A + ridge * I
    rhs = A' * B

    coef = R \ rhs

    return vec(coef[1, :])
end

function predict_next(model, x::AbstractVector{<:Real})
    x_vec = Float64.(collect(x))

    x_now_n = standardize_vector(x_vec, model.x_mean, model.x_std)

    y_next_n = local_linear_predict(
        x_now_n,
        model.Xn,
        model.Yn,
        model.k_neighbors,
        model.ridge,
        model.bandwidth_factor
    )

    y_next = destandardize_vector(y_next_n, model.y_mean, model.y_std)

    return y_next
end

function iterate_map(model, x0::AbstractVector{<:Real}, n::Integer)
    x0_vec = Float64.(collect(x0))
    nx = length(x0_vec)

    orbit = zeros(n, nx)
    orbit[1, :] .= x0_vec

    for k in 1:(n-1)
        orbit[k+1, :] .= predict_next(model, vec(orbit[k, :]))
    end

    return orbit
end

end
