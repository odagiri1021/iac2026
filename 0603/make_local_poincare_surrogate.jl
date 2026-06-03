using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Serialization

# ============================================================
# settings
# ============================================================

script_dir = @__DIR__

csv_path = joinpath(
    script_dir,
    "plots_ganimede_background",
    "background_poincare_Crow_Vern9_section_points.csv"
)

# 入力に使う断面座標
# 基本: y=0断面で vyの符号を固定しているなら [:x, :vx] でよい
input_cols = [:x, :vx]

# 出力として予測する次の断面座標
output_cols = [:x, :vx]

group_col = :seed_id
sort_col = :hit

# vy < 0 の断面だけ使うか
# CSV作成時点ですでに vy < 0 だけなら、このまま true でも問題ない
use_vy_filter = true
vy_col = :vy
vy_sign = -1.0       # vy < 0なら -1.0, vy > 0なら +1.0
vy_eps = 0.0

# 局所モデル設定
k_neighbors = 80     # 近傍点数。小さいほど局所的、大きいほど滑らか
ridge = 1.0e-8       # 局所線形回帰の正則化
bandwidth_factor = 1.0

# 反復計算設定
num_iter = 1000

model_save_path = joinpath(script_dir, "local_poincare_model.jls")
orbit_save_path = joinpath(script_dir, "local_predicted_iterated_orbit.csv")
plot_save_path = joinpath(script_dir, "local_poincare_surrogate.png")

# ============================================================
# utility
# ============================================================

function standardize_matrix(X, μ, σ)
    Xn = similar(X)
    for j in 1:size(X, 2)
        Xn[:, j] = (X[:, j] .- μ[j]) ./ σ[j]
    end
    return Xn
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

function make_pairs(df, input_cols, output_cols, group_col, sort_col)
    X_list = Vector{Vector{Float64}}()
    Y_list = Vector{Vector{Float64}}()

    gdf = groupby(df, group_col)

    for subdf0 in gdf
        subdf = sort(subdf0, sort_col)

        if nrow(subdf) < 2
            continue
        end

        for i in 1:(nrow(subdf)-1)
            x_now = [Float64(subdf[i, c]) for c in input_cols]
            x_next = [Float64(subdf[i+1, c]) for c in output_cols]

            push!(X_list, x_now)
            push!(Y_list, x_next)
        end
    end

    N = length(X_list)
    nx = length(input_cols)
    ny = length(output_cols)

    X = zeros(N, nx)
    Y = zeros(N, ny)

    for i in 1:N
        X[i, :] .= X_list[i]
        Y[i, :] .= Y_list[i]
    end

    return X, Y
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

    # 局所線形:
    # Yn ≈ [1, dx1, dx2, ...] * B
    # query点では dx = 0 なので、予測値は B[1, :]
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

    y_query_n = vec(coef[1, :])

    return y_query_n
end

# ============================================================
# load CSV
# ============================================================

df = CSV.read(csv_path, DataFrame)

println("Loaded: ", csv_path)
println("Columns: ", names(df))
println("Number of rows: ", nrow(df))

required_cols = unique(vcat(input_cols, output_cols, [group_col, sort_col]))

for c in required_cols
    if !(string(c) in names(df))
        error("CSVに必要な列がありません: ", c)
    end
end

if use_vy_filter
    if !(string(vy_col) in names(df))
        error("use_vy_filter = true ですが、CSVに vy列 がありません。")
    end

    if vy_sign < 0
        df = df[df[!, vy_col] .< -vy_eps, :]
        println("Filtered by vy < ", -vy_eps)
    else
        df = df[df[!, vy_col] .> vy_eps, :]
        println("Filtered by vy > ", vy_eps)
    end

    println("Rows after vy filter: ", nrow(df))
end

df = dropmissing(df, required_cols)

valid = trues(nrow(df))
for c in required_cols
    valid .&= isfinite.(Float64.(df[!, c]))
end
df = df[valid, :]

println("Valid rows: ", nrow(df))

# ============================================================
# make map pairs
# ============================================================

X, Y = make_pairs(df, input_cols, output_cols, group_col, sort_col)

println("Number of map pairs: ", size(X, 1))
println("Input columns : ", input_cols)
println("Output columns: ", output_cols)

if size(X, 1) < k_neighbors
    error("写像ペア数が k_neighbors より少ないです。k_neighbors を小さくしてください。")
end

# ============================================================
# standardize
# ============================================================

x_mean = vec(mean(X, dims=1))
x_std = vec(std(X, dims=1))
y_mean = vec(mean(Y, dims=1))
y_std = vec(std(Y, dims=1))

x_std[x_std .== 0.0] .= 1.0
y_std[y_std .== 0.0] .= 1.0

Xn = standardize_matrix(X, x_mean, x_std)
Yn = standardize_matrix(Y, y_mean, y_std)

# ============================================================
# save model database
# ============================================================

model = (
    input_cols = input_cols,
    output_cols = output_cols,
    group_col = group_col,
    sort_col = sort_col,
    use_vy_filter = use_vy_filter,
    vy_sign = vy_sign,
    k_neighbors = k_neighbors,
    ridge = ridge,
    bandwidth_factor = bandwidth_factor,
    x_mean = x_mean,
    x_std = x_std,
    y_mean = y_mean,
    y_std = y_std,
    X = X,
    Y = Y,
    Xn = Xn,
    Yn = Yn
)

serialize(model_save_path, model)
println("Saved model to: ", model_save_path)

# ============================================================
# iterate local surrogate
# ============================================================

x0 = vec(X[1, :])

orbit_pred = zeros(num_iter, length(input_cols))
orbit_pred[1, :] .= x0

for k in 1:(num_iter-1)
    x_now = vec(orbit_pred[k, :])
    x_now_n = standardize_vector(x_now, x_mean, x_std)

    y_next_n = local_linear_predict(
        x_now_n,
        Xn,
        Yn,
        k_neighbors,
        ridge,
        bandwidth_factor
    )

    y_next = destandardize_vector(y_next_n, y_mean, y_std)
    orbit_pred[k+1, :] .= y_next
end

out_df = DataFrame()
for j in 1:length(output_cols)
    out_df[!, output_cols[j]] = orbit_pred[:, j]
end

CSV.write(orbit_save_path, out_df)
println("Saved iterated local surrogate orbit to: ", orbit_save_path)

# ============================================================
# plot
# ============================================================

try
    using CairoMakie

    fig = Figure(size = (900, 700))
    ax = Axis(
        fig[1, 1],
        xlabel = string(input_cols[1]),
        ylabel = string(input_cols[2]),
        title = "Local Poincare surrogate"
    )

    scatter!(
        ax,
        X[:, 1],
        X[:, 2],
        markersize = 2,
        label = "original section points"
    )

    lines!(
        ax,
        orbit_pred[:, 1],
        orbit_pred[:, 2],
        linewidth = 1,
        label = "local surrogate iterated"
    )

    axislegend(ax)
    save(plot_save_path, fig)

    println("Saved plot to: ", plot_save_path)
catch e
    println("Plot skipped. CairoMakie may not be installed.")
    println(e)
end
