using DifferentialEquations
using LinearAlgebra
using GLMakie
using CSV
using DataFrames
using Printf
using Base.Threads

# ==============================================================================
# 1. パラメータ
# ==============================================================================
# 注意: ここを木星-ガニメデ系の質量比にする（例: 7.80655e-5）
const MU       = 7.806548101361265e-05
const C_TARGET = 3.0068
const ORDER0   = 2
const ORDERF   = 7
const SIGN_V   = -1.0

const X0_START = 0.99
const XF_END   = 0.95
const N_POINTS = 5000

const MAX_ITER   = 100
const JUDGE_CONV = 1e-12

# 積分時間上限（元コードに合わせて 50）
const TMAX = 50.0

# 重要: スキャンとNewtonで許容誤差を分ける（ここが効く）
const SCAN_RELTOL   = 1e-11
const SCAN_ABSTOL   = 1e-12
const NEWTON_RELTOL = 3e-14
const NEWTON_ABSTOL = 1e-14

# 安定判定の閾値（Phi(T) の非自明固有値の |λ| が 1+EPS を超えたら不安定）
const STABILITY_EPS = 1e-3

# 保存先
const BASE_DIR = raw"C:\Users\tmyko\OneDrive - Kyushu University\lab_stochastic_optimal_control\periodic_orbit"
const PLOT_DIR = joinpath(BASE_DIR, "plots_ganimede")
const CSV_PATH = joinpath(BASE_DIR, "found_orbits_ganimede.csv")
mkpath(PLOT_DIR)

# ==============================================================================
# order -> (p,q) の対応（自分の定義で埋める）
#   例: order=5 が 7:5 なら P_BY_ORDER[5]=7, Q_BY_ORDER[5]=5
# ==============================================================================

# --- 例（仮）必要に応じて書き換え ---
# P_BY_ORDER[2]=3; Q_BY_ORDER[2]=2
# P_BY_ORDER[3]=4; Q_BY_ORDER[3]=3
# P_BY_ORDER[4]=5; Q_BY_ORDER[4]=4
# P_BY_ORDER[5]=7; Q_BY_ORDER[5]=5
# P_BY_ORDER[6]=8; Q_BY_ORDER[6]=6
# P_BY_ORDER[7]=9; Q_BY_ORDER[7]=7

# ==============================================================================
# p:q を自動で作る（ファイル名は ":" を使えないので p07q05 形式）
# ==============================================================================
@inline function pq_from_period(order::Int, T::Float64)
    p = order
    q_real = T / (2π)
    q = max(1, round(Int, q_real))
    pq_str = @sprintf("%d:%d", p, q)         # タイトル用
    pq_tag = @sprintf("p%02dq%02d", p, q)    # ファイル名用
    q_err  = abs(q_real - q)                # どれだけ整数からズレてるか（確認用）
    return p, q, pq_str, pq_tag, q_err
end


# ==============================================================================
# 2. 型固定パラメータ
# ==============================================================================
struct CR3BPParams
    μ::Float64
    C::Float64
    sign_v::Float64
end
const P = CR3BPParams(MU, C_TARGET, SIGN_V)

# ==============================================================================
# 3. 基本関数（vy0計算）
# ==============================================================================
@inline function vy0_from_jacobi(x0::Float64, p::CR3BPParams)
    μ = p.μ
    r1 = abs(x0 + μ)
    r2 = abs(x0 - 1 + μ)
    U  = 0.5*x0*x0 + (1-μ)/r1 + μ/r2
    v2 = 2*U - p.C
    if v2 <= 0
        return NaN
    end
    return p.sign_v * sqrt(v2)
end

# ==============================================================================
# 4. CR3BP 4次元（状態のみ）: 速い右辺（アロケーションほぼゼロ）
# ==============================================================================
function cr3bp_dynamics!(du, u, p::CR3BPParams, t)
    @inbounds begin
        μ  = p.μ
        x  = u[1]
        y  = u[2]
        vx = u[3]
        vy = u[4]

        r1sq = (x + μ)*(x + μ) + y*y
        r2sq = (x - 1 + μ)*(x - 1 + μ) + y*y

        r1   = sqrt(r1sq)
        r2   = sqrt(r2sq)

        inv_r1_3 = 1.0/(r1sq*r1)
        inv_r2_3 = 1.0/(r2sq*r2)

        om = 1.0 - μ

        du[1] = vx
        du[2] = vy
        du[3] = 2*vy + x - om*(x + μ)*inv_r1_3 - μ*(x - 1 + μ)*inv_r2_3
        du[4] = -2*vx + y - om*y*inv_r1_3 - μ*y*inv_r2_3
    end
    return nothing
end

# ==============================================================================
# 5. CR3BP + STM 20次元: 行列を作らず手で回す（ここが重要）
# ==============================================================================
const PHI0_VEC = let M = Matrix{Float64}(I, 4, 4); vec(M) end

function cr3bp_stm_dynamics!(du, u, p::CR3BPParams, t)
    @inbounds begin
        μ  = p.μ
        x  = u[1]
        y  = u[2]
        vx = u[3]
        vy = u[4]

        r1sq = (x + μ)*(x + μ) + y*y
        r2sq = (x - 1 + μ)*(x - 1 + μ) + y*y

        r1   = sqrt(r1sq)
        r2   = sqrt(r2sq)

        inv_r1_3 = 1.0/(r1sq*r1)
        inv_r2_3 = 1.0/(r2sq*r2)

        inv_r1_5 = inv_r1_3/r1sq
        inv_r2_5 = inv_r2_3/r2sq

        om = 1.0 - μ

        # state
        du[1] = vx
        du[2] = vy
        du[3] = 2*vy + x - om*(x + μ)*inv_r1_3 - μ*(x - 1 + μ)*inv_r2_3
        du[4] = -2*vx + y - om*y*inv_r1_3 - μ*y*inv_r2_3

        # Jacobian pieces
        Ωxx = 1 - om*inv_r1_3 + 3*om*(x+μ)*(x+μ)*inv_r1_5 - μ*inv_r2_3 + 3*μ*(x-1+μ)*(x-1+μ)*inv_r2_5
        Ωxy = 3*om*(x+μ)*y*inv_r1_5 + 3*μ*(x-1+μ)*y*inv_r2_5
        Ωyy = 1 - om*inv_r1_3 + 3*om*y*y*inv_r1_5 - μ*inv_r2_3 + 3*μ*y*y*inv_r2_5

        # STM: Φ_dot = A*Φ を列ごとに手で更新
        base = 5
        for j in 0:3
            k  = base + 4*j
            φ1 = u[k]
            φ2 = u[k+1]
            φ3 = u[k+2]
            φ4 = u[k+3]

            du[k]   = φ3
            du[k+1] = φ4
            du[k+2] = Ωxx*φ1 + Ωxy*φ2 + 2*φ4
            du[k+3] = Ωxy*φ1 + Ωyy*φ2 - 2*φ3
        end
    end
    return nothing
end

# ==============================================================================
# 6. セクションカウント用Callback
# ==============================================================================
mutable struct SectionCounter
    count::Int
    target::Int
end

function make_y0_counter_callback(sec::SectionCounter)
    condition(u, t, integrator) = u[2]  # y = 0
    function affect!(integrator)
        if integrator.t == 0.0
            return
        end
        sec.count += 1
        if sec.count >= sec.target
            terminate!(integrator)
        end
    end
    return ContinuousCallback(condition, affect!; save_positions=(false, false))
end

function make_abort_callback()
    condition(u, t, integrator) = (abs(u[1]) > 5.0) || (abs(u[2]) > 5.0)
    affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

# ==============================================================================
# 7. 4次元シューティング（スキャン用）
# ==============================================================================
function shooting_4d!(u0::Vector{Float64},
                      x0::Float64,
                      order::Int,
                      p::CR3BPParams,
                      sec::SectionCounter,
                      cb_counter,
                      cb_abort;
                      reltol::Float64,
                      abstol::Float64)

    vy0 = vy0_from_jacobi(x0, p)
    if !isfinite(vy0)
        return (NaN, NaN, NaN, nothing)
    end

    u0[1] = x0
    u0[2] = 0.0
    u0[3] = 0.0
    u0[4] = vy0

    sec.count  = 0
    sec.target = order

    prob = ODEProblem(cr3bp_dynamics!, u0, (0.0, TMAX), p)
    cb   = CallbackSet(cb_counter, cb_abort)
    sol  = solve(prob, Vern9(); callback=cb, reltol=reltol, abstol=abstol, save_everystep=false)

    if sec.count < order
        return (NaN, NaN, NaN, nothing)
    end

    xf = sol.u[end]
    return (xf[3], xf[4], sol.t[end], xf)  # vx_f, vy_f, T_half, xf
end

# ==============================================================================
# 8. STMシューティング（Newton用: res と dres と Thalf）
# ==============================================================================
function shooting_stm!(u0::Vector{Float64},
                       x0::Float64,
                       order::Int,
                       p::CR3BPParams,
                       sec::SectionCounter,
                       cb_counter,
                       cb_abort;
                       reltol::Float64,
                       abstol::Float64)

    vy0 = vy0_from_jacobi(x0, p)
    if !isfinite(vy0)
        return (NaN, NaN, NaN)
    end

    u0[1] = x0
    u0[2] = 0.0
    u0[3] = 0.0
    u0[4] = vy0
    @inbounds for i in 1:16
        u0[4+i] = PHI0_VEC[i]
    end

    sec.count  = 0
    sec.target = order

    prob = ODEProblem(cr3bp_stm_dynamics!, u0, (0.0, TMAX), p)
    cb   = CallbackSet(cb_counter, cb_abort)
    sol  = solve(prob, Vern9(); callback=cb, reltol=reltol, abstol=abstol, save_everystep=false)

    if sec.count < order
        return (NaN, NaN, NaN)
    end

    xf = sol.u[end]
    vx_f = xf[3]
    vy_f = xf[4]
    Thalf = sol.t[end]

    if !isfinite(vy_f) || abs(vy_f) < 1e-14
        return (NaN, NaN, NaN)
    end

    # dvy0/dx0
    μ = p.μ
    r1_0 = abs(x0 + μ)
    r2_0 = abs(x0 - 1 + μ)
    dU0_dx0 = x0 - (1-μ)*(x0+μ)/(r1_0^3) - μ*(x0-1+μ)/(r2_0^3)
    dvy0_dx0 = dU0_dx0 / vy0

    # Φ の必要成分だけ取り出す（column-major, base=5）
    # Φ21, Φ24, Φ31, Φ34
    Φ21 = xf[6]
    Φ31 = xf[7]
    Φ24 = xf[18]
    Φ34 = xf[19]

    # vx_dot_f
    x = xf[1]; y = xf[2]
    r1sq = (x + μ)*(x + μ) + y*y
    r2sq = (x - 1 + μ)*(x - 1 + μ) + y*y
    r1   = sqrt(r1sq)
    r2   = sqrt(r2sq)
    inv_r1_3 = 1.0/(r1sq*r1)
    inv_r2_3 = 1.0/(r2sq*r2)
    vx_dot_f = 2*vy_f + x - (1-μ)*(x+μ)*inv_r1_3 - μ*(x-1+μ)*inv_r2_3

    # dres/dx0
    dres = (Φ31 + Φ34*dvy0_dx0) - (vx_dot_f / vy_f) * (Φ21 + Φ24*dvy0_dx0)

    return (vx_f, dres, Thalf)
end

# ==============================================================================
# 9. ブラケット付きNewton
# ==============================================================================
function refine_root_newton_bisect(xl::Float64, xr::Float64, fl::Float64, fr::Float64,
                                  order::Int, p::CR3BPParams;
                                  max_iter::Int = MAX_ITER)

    sec = SectionCounter(0, order)
    cb_counter = make_y0_counter_callback(sec)
    cb_abort   = make_abort_callback()

    u0 = zeros(20)
    x = 0.5*(xl + xr)
    Thalf_best = NaN

    for it in 1:max_iter
        res, dres, Thalf = shooting_stm!(u0, x, order, p, sec, cb_counter, cb_abort;
                                         reltol=NEWTON_RELTOL, abstol=NEWTON_ABSTOL)
        if !isfinite(res) || !isfinite(dres) || abs(dres) < 1e-18
            x = 0.5*(xl + xr)
            res, dres, Thalf = shooting_stm!(u0, x, order, p, sec, cb_counter, cb_abort;
                                             reltol=NEWTON_RELTOL, abstol=NEWTON_ABSTOL)
            if !isfinite(res)
                return (false, NaN, NaN)
            end
        end

        Thalf_best = Thalf

        if abs(res) < JUDGE_CONV
            return (true, x, Thalf_best)
        end

        if res*fl < 0
            xr = x
            fr = res
        else
            xl = x
            fl = res
        end

        x_new = x - res/dres
        if !isfinite(x_new) || (x_new <= min(xl, xr)) || (x_new >= max(xl, xr))
            x_new = 0.5*(xl + xr)
        end
        x = x_new
    end

    return (false, NaN, NaN)
end

# ==============================================================================
# 10. 安定判定: Phi(T) の固有値から stable/unstable を決める
# ==============================================================================
function classify_stability(x0::Float64, T::Float64, p::CR3BPParams)
    vy0 = vy0_from_jacobi(x0, p)
    if !isfinite(vy0) || !(T > 0)
        return (false, NaN)
    end

    u0 = zeros(20)
    u0[1] = x0
    u0[2] = 0.0
    u0[3] = 0.0
    u0[4] = vy0
    @inbounds for i in 1:16
        u0[4+i] = PHI0_VEC[i]
    end

    prob = ODEProblem(cr3bp_stm_dynamics!, u0, (0.0, T), p)
    sol  = solve(prob, Vern9(); reltol=NEWTON_RELTOL, abstol=NEWTON_ABSTOL, save_everystep=false)

    xf = sol.u[end]
    Φ = reshape(@view(xf[5:20]), 4, 4)  # column-major

    λ = eigvals(Matrix(Φ))  # 4 eigenvalues (possibly complex)

    # 2つの自明固有値(ほぼ1)を除外: |λ-1| が小さい順に2個を落とす
    idx = sortperm(abs.(λ .- 1.0))
    trivial = idx[1:2]
    others = Int[]
    for k in 1:4
        if !(k in trivial)
            push!(others, k)
        end
    end

    ρ = maximum(abs.(λ[others]))
    stable = ρ <= (1.0 + STABILITY_EPS)

    return (stable, ρ)
end

function run_search()
    xspan = collect(range(X0_START, XF_END, length=N_POINTS))

    results_df = DataFrame(
        ID=Int[],
        x0=Float64[], y0=Float64[], vx0=Float64[], vy0=Float64[],
        T=Float64[], order=Int[],
        p=Int[], q=Int[], q_err=Float64[],
        stable=Bool[], rho=Float64[]
    )
    found_x_list = Float64[]

    println("Search start: C=$(C_TARGET), N=$(N_POINTS), threads=$(nthreads())")
    println("Scan tol: rel=$(SCAN_RELTOL), abs=$(SCAN_ABSTOL)")
    println("Newton tol: rel=$(NEWTON_RELTOL), abs=$(NEWTON_ABSTOL)")
    println("MU=$(MU)")

    for order in ORDER0:ORDERF
        println("order = $order")

        # ---- まず 4D で全点スキャン（並列）
        res = fill(NaN, length(xspan))
        Th  = fill(NaN, length(xspan))

        nt = nthreads()
        u0s  = [zeros(4) for _ in 1:nt]
        secs = [SectionCounter(0, order) for _ in 1:nt]
        cbs  = [make_y0_counter_callback(secs[k]) for k in 1:nt]
        cab  = [make_abort_callback() for _ in 1:nt]

        @threads for i in eachindex(xspan)
            tid = threadid()
            vx_f, _, Thalf, _ = shooting_4d!(u0s[tid], xspan[i], order, P, secs[tid], cbs[tid], cab[tid];
                                             reltol=SCAN_RELTOL, abstol=SCAN_ABSTOL)
            res[i] = vx_f
            Th[i]  = Thalf
        end

        # ---- 符号反転区間を抽出
        brackets = Tuple{Int,Int}[]
        for i in 1:(length(xspan)-1)
            a = res[i]
            b = res[i+1]
            if isfinite(a) && isfinite(b) && a*b < 0
                push!(brackets, (i, i+1))
            end
        end
        println("Brackets found: $(length(brackets))")

        # ---- 各ブラケットをNewton+二分法で収束
        for (ia, ib) in brackets
            xa = xspan[ia]
            xb = xspan[ib]
            fa = res[ia]
            fb = res[ib]

            ok, x_root, Thalf = refine_root_newton_bisect(xa, xb, fa, fb, order, P; max_iter=MAX_ITER)
            if !ok || !isfinite(x_root) || !isfinite(Thalf)
                continue
            end

            # 重複除去
            if any(isapprox.(found_x_list, x_root, atol=1e-6))
                continue
            end
            push!(found_x_list, x_root)

            # 初期条件確定
            y0  = 0.0
            vx0 = 0.0
            vy0 = vy0_from_jacobi(x_root, P)
            if !isfinite(vy0)
                continue
            end

            T  = 2.0*Thalf

            # ---- ここで p:q を作る（対応表は使わない）
            p_res, q_res, pq_str, pq_tag, q_err = pq_from_period(order, T)

            # 安定判定
            stable, rho = classify_stability(x_root, T, P)

            id = length(found_x_list)
            push!(results_df, (id, x_root, y0, vx0, vy0, T, order, p_res, q_res, q_err, stable, rho))

            # 軌道描画（色分け）
            u0_plot = [x_root, y0, vx0, vy0]
            prob_p = ODEProblem(cr3bp_dynamics!, u0_plot, (0.0, T), P)
            tgrid = range(0.0, T; length=3000)
            sol_p = solve(prob_p, Vern9(); reltol=SCAN_RELTOL, abstol=SCAN_ABSTOL, saveat=tgrid)

            fig = Figure(size=(1000, 800))
            title_str = @sprintf("ID:%d  p:q=%s  x0=%.10f  C=%.4f  order=%d  %s  rho=%.6f  q_err=%.3e",
                                 id, pq_str, x_root, C_TARGET, order,
                                 stable ? "stable" : "unstable", rho, q_err)
            ax = Axis(fig[1,1], title=title_str, aspect=DataAspect())

            col = stable ? :blue : :red
            lines!(ax, sol_p[1,:], sol_p[2,:]; linewidth=2, color=col)
            scatter!(ax, [-MU, 1-MU], [0, 0]; markersize=15)

            xlims!(ax, -1.1, 1.1)
            ylims!(ax, -1.1, 1.1)

            outname = @sprintf("orbit_%03d_%s_%s.png", id, pq_tag, stable ? "stable" : "unstable")
            save(joinpath(PLOT_DIR, outname), fig)

            println(@sprintf("Found ID %d: p:q=%s  x0=%.12f  T=%.12f  %s  rho=%.6f  q_err=%.3e  file=%s",
                             id, pq_str, x_root, T, stable ? "stable" : "unstable", rho, q_err, outname))

        end
    end

    CSV.write(CSV_PATH, results_df)
    println("Done. CSV: $CSV_PATH  plots: $PLOT_DIR")
end

run_search()