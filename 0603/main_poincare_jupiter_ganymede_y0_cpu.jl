# ============================================================
# Background Poincare map: KAM tori / chaotic sea / resonant islands
# Based on GanimedePoincareManifolds style.
#
# 島・トーラス構造を見るには「少数 seed × 長時間積分」が有効。
#   - nx=20, nvx=20 + Smax=1200, tmax=8000 で各軌道をじっくり追う
#   - KAM曲線 → 断面上の閉じた点列
#   - 共鳴島 → 複数の孤立した点群 (island chain)
#   - カオス海 → 広く拡散した点群
# ============================================================

module GanimedePoincareBackground
using SciMLBase: ContinuousCallback, CallbackSet, terminate!
using OrdinaryDiffEqVerner: Vern9

using LinearAlgebra
using Printf
using CairoMakie
using DataFrames
using CSV
using SciMLBase

CairoMakie.activate!(type = "png")

# ==============================================================================
# 0. 定数・パス (マニフォールドコードと統一)
# ==============================================================================
const MU = 7.806548101361265e-05

const ROW = (
    x0     = 0.958000406,
    y0     = 0.0,
    vx0    = 0.0,
    vy0    = -0.045142641,
    T      = 31.58106166,
    p      = 3,
    q      = 5,
)

const BASE_DIR = @__DIR__
const PLOT_DIR = joinpath(BASE_DIR, "plots_ganimede_background")
mkpath(PLOT_DIR)

# ==============================================================================
# 1. CR3BP パラメータ構造体
# ==============================================================================
struct CR3BPParams
    μ::Float64
end

const P = CR3BPParams(MU)

# ==============================================================================
# 2. ヤコビ定数
# ==============================================================================
function jacobi_constant(u::AbstractVector{<:Real}, μ::Real)
    x, y, vx, vy = u[1], u[2], u[3], u[4]
    r1 = sqrt((x + μ)^2 + y^2)
    r2 = sqrt((x - 1 + μ)^2 + y^2)
    Ω  = 0.5*(x^2 + y^2) + (1 - μ)/r1 + μ/r2
    return 2Ω - vx^2 - vy^2
end

# ROW の初期値から C を計算
const C_ROW = jacobi_constant([ROW.x0, ROW.y0, ROW.vx0, ROW.vy0], MU)

# ==============================================================================
# 3. Config
# ==============================================================================
Base.@kwdef struct Config
    C::Float64 = C_ROW

    # ---- seed グリッド ----
    nx::Int    = 20          # x 方向分割数 (少数 seed 戦略)
    nvx::Int   = 20          # ẋ 方向分割数

    x_min_seed::Float64   = -0.95
    x_max_seed::Float64   = -0.70
    vx_min_seed::Float64  = -0.18
    vx_max_seed::Float64  =  0.18

    # 初期 ẏ の符号 (:pos / :neg / :both)
    initial_vy_branch::Symbol = :pos

    # ---- 断面フィルタ ----
    xneg::Bool            = true
    vy_filter::Symbol     = :none   # :none=両方向, :pos, :neg

    # ---- 積分設定 ----
    # 「少数 seed × 長時間」で島・トーラス構造を鮮明に
    Smax::Int             = 1200    # 1 seed あたりの最大交差数
    tmax::Float64         = 8000.0  # 最大積分時間

    reltol::Float64       = 1e-12
    abstol::Float64       = 1e-12
    abort_limit::Float64  = 10.0

    # 衝突回避 (木星半径≈0.0668, ガニメデ半径≈0.00246)
    r1_min::Float64       = 0.070
    r2_min::Float64       = 0.005

    # ヤコビ定数のドリフト許容量 (数値誤差チェック)
    C_error_keep::Float64 = 1e-7

    # ---- プロット ----
    xlim::Tuple{Float64,Float64} = (-1.20, -0.50)
    ylim::Tuple{Float64,Float64} = (-0.35,  0.35)
    markersize::Float64          = 0.8

    # ---- 出力 ----
    save_csv::Bool   = true
    save_png::Bool   = true
    file_stub::String = "background_poincare_islands"
end

# ==============================================================================
# 4. 運動方程式 (マニフォールドコードと同一)
# ==============================================================================
function cr3bp_dynamics!(du, u, p::CR3BPParams, t)
    @inbounds begin
        μ  = p.μ
        x, y, vx, vy = u[1], u[2], u[3], u[4]

        r1sq = (x + μ)^2 + y^2
        r2sq = (x - 1 + μ)^2 + y^2
        r1   = sqrt(r1sq);  r2 = sqrt(r2sq)
        ir1  = 1/(r1sq*r1); ir2 = 1/(r2sq*r2)
        om   = 1 - μ

        du[1] = vx
        du[2] = vy
        du[3] =  2vy + x - om*(x + μ)*ir1 - μ*(x - 1 + μ)*ir2
        du[4] = -2vx + y - om*y*ir1        - μ*y*ir2
    end
    return nothing
end

# ==============================================================================
# 5. 断面上の ẏ を C から計算
# ==============================================================================
function vy_from_C_on_section(x::Float64, vx::Float64,
                               C::Float64, μ::Float64,
                               branch::Symbol)
    r1  = abs(x + μ)
    r2  = abs(x - 1 + μ)
    Ω   = 0.5*x^2 + (1-μ)/r1 + μ/r2
    vy2 = 2Ω - C - vx^2
    vy2 < 0 && return false, NaN
    vy_abs = sqrt(max(vy2, 0.0))
    branch === :pos && return true,  vy_abs
    branch === :neg && return true, -vy_abs
    error("branch は :pos か :neg")
end

# ==============================================================================
# 6. seed グリッド生成
# ==============================================================================
function make_seed_grid(conf::Config)
    seeds   = Vector{Vector{Float64}}()
    seed_x  = Float64[]
    seed_vx = Float64[]

    xs  = range(conf.x_min_seed,  conf.x_max_seed;  length=conf.nx)
    vxs = range(conf.vx_min_seed, conf.vx_max_seed; length=conf.nvx)

    branches = conf.initial_vy_branch === :both  ? (:pos, :neg) :
               conf.initial_vy_branch === :pos   ? (:pos,)      :
               conf.initial_vy_branch === :neg   ? (:neg,)      :
               error("initial_vy_branch は :pos/:neg/:both")

    for x in xs
        conf.xneg && x >= 0 && continue
        for vx in vxs
            for br in branches
                ok, vy = vy_from_C_on_section(Float64(x), Float64(vx),
                                               conf.C, MU, br)
                ok || continue
                push!(seeds,   [Float64(x), 0.0, Float64(vx), vy])
                push!(seed_x,  Float64(x))
                push!(seed_vx, Float64(vx))
            end
        end
    end
    return seeds, seed_x, seed_vx
end

# ==============================================================================
# 7. ポアンカレ断面交差収集 (マニフォールドコードと同スタイル)
# ==============================================================================
function collect_poincare_hits(u0::Vector{Float64},
                                p::CR3BPParams;
                                tspan::Tuple{Float64,Float64},
                                Smax::Int         = 1200,
                                xneg::Bool        = true,
                                vy_filter::Symbol = :none,
                                reltol::Float64   = 1e-12,
                                abstol::Float64   = 1e-12,
                                abort_limit::Float64 = 10.0,
                                r1_min::Float64   = 0.070,
                                r2_min::Float64   = 0.005,
                                C_ref::Float64    = NaN,
                                C_error_keep::Float64 = 1e-7,
                                t_ignore::Float64 = 1e-10,
                                dt_min::Float64   = 1e-8)

    ts  = Float64[]; xs  = Float64[]; ys  = Float64[]
    vxs = Float64[]; vys = Float64[]
    last_t = Ref(NaN)

    condition(u, t, integrator) = u[2]

    function affect!(integrator)
        t = integrator.t
        abs(t) < t_ignore && return
        !isnan(last_t[]) && abs(t - last_t[]) < dt_min && return

        x, y, vx, vy = integrator.u[1], integrator.u[2],
                        integrator.u[3], integrator.u[4]
        μ = p.μ
        r1 = sqrt((x + μ)^2 + y^2)
        r2 = sqrt((x - 1 + μ)^2 + y^2)

        # 衝突回避
        (r1 < r1_min || r2 < r2_min) && (terminate!(integrator); return)

        # ヤコビ定数ドリフトチェック
        if isfinite(C_ref)
            C_now = jacobi_constant([x, y, vx, vy], μ)
            abs(C_now - C_ref) > C_error_keep && return
        end

        # 断面フィルタ
        xneg && x >= 0 && return
        vy_filter === :pos && vy <= 0 && return
        vy_filter === :neg && vy >= 0 && return

        last_t[] = t
        push!(ts, t); push!(xs, x); push!(ys, y)
        push!(vxs, vx); push!(vys, vy)

        length(xs) >= Smax && terminate!(integrator)
        return
    end

    cb_cross = ContinuousCallback(condition, affect!;
                                  save_positions=(false, false))
    cb_abort = DiscreteCallback(
        (u, t, int) -> begin
            μ = p.μ
            r1 = sqrt((u[1]+μ)^2 + u[2]^2)
            r2 = sqrt((u[1]-1+μ)^2 + u[2]^2)
            abs(u[1]) > abort_limit || abs(u[2]) > abort_limit ||
            r1 < r1_min || r2 < r2_min
        end,
        int -> terminate!(int);
        save_positions=(false, false))

    prob = ODEProblem(cr3bp_dynamics!, u0, tspan, p)
    solve(prob, Vern9();
          callback       = CallbackSet(cb_cross, cb_abort),
          reltol         = reltol,
          abstol         = abstol,
          save_everystep = false)

    return ts, xs, ys, vxs, vys
end

# ==============================================================================
# 8. 背景計算メイン
# ==============================================================================
function compute_background(conf::Config = Config())
    seeds, seed_x, seed_vx = make_seed_grid(conf)

    @printf("C  = %.16e\n", conf.C)
    @printf("有効 seed 数 = %d  (格子 %d×%d)\n",
            length(seeds), conf.nx, conf.nvx)
    @printf("Smax = %d, tmax = %.1f\n", conf.Smax, conf.tmax)
    @printf("vy_filter = %s, initial_vy_branch = %s\n",
            String(conf.vy_filter), String(conf.initial_vy_branch))

    seed_id_all = Int[];  hit_id_all = Int[]
    t_all       = Float64[]; x_all = Float64[]
    y_all       = Float64[]; vx_all = Float64[]
    vy_all      = Float64[]; C_all  = Float64[]

    for sid in eachindex(seeds)
        (sid == 1 || sid % 20 == 0 || sid == length(seeds)) &&
            @printf("  seed %4d / %4d\n", sid, length(seeds))
        flush(stdout)

        ts, xs, ys, vxs, vys = collect_poincare_hits(
            seeds[sid], P;
            tspan         = (0.0, conf.tmax),
            Smax          = conf.Smax,
            xneg          = conf.xneg,
            vy_filter     = conf.vy_filter,
            reltol        = conf.reltol,
            abstol        = conf.abstol,
            abort_limit   = conf.abort_limit,
            r1_min        = conf.r1_min,
            r2_min        = conf.r2_min,
            C_ref         = conf.C,
            C_error_keep  = conf.C_error_keep,
        )

        for k in eachindex(xs)
            uhit = [xs[k], ys[k], vxs[k], vys[k]]
            push!(seed_id_all, sid);  push!(hit_id_all, k)
            push!(t_all,  ts[k]);     push!(x_all,  xs[k])
            push!(y_all,  ys[k]);     push!(vx_all, vxs[k])
            push!(vy_all, vys[k])
            push!(C_all,  jacobi_constant(uhit, MU))
        end
    end

    df = DataFrame(
        seed_id = seed_id_all,
        hit     = hit_id_all,
        t       = t_all,
        x       = x_all,
        y       = y_all,
        vx      = vx_all,
        vy      = vy_all,
        C       = C_all,
        C_error = C_all .- conf.C,
    )

    seed_df = DataFrame(
        seed_id = 1:length(seeds),
        x0      = seed_x,
        vx0     = seed_vx,
        vy0     = [s[4] for s in seeds],
    )

    return df, seed_df
end

# ==============================================================================
# 9. プロット
# ==============================================================================
function plot_background(df::DataFrame,
                         seed_df::DataFrame,
                         conf::Config = Config();
                         plot_seeds::Bool = false)

    fig = Figure(size=(1200, 900))
    ax  = Axis(fig[1, 1],
               xlabel = "x [-]",
               ylabel = "ẋ [-]",
               title  = @sprintf(
                   "Background Poincaré section  C=%.10f  crossings=%d",
                   conf.C, nrow(df)))

    nrow(df) > 0 && scatter!(ax, df.x, df.vx;
                              markersize = conf.markersize,
                              color      = RGBf(0.0, 0.0, 0.0))

    plot_seeds && nrow(seed_df) > 0 &&
        scatter!(ax, seed_df.x0, seed_df.vx0;
                 markersize = 0.8,
                 color      = RGBf(0.85, 0.0, 0.0))

    xlims!(ax, conf.xlim...)
    ylims!(ax, conf.ylim...)
    return fig
end

# ==============================================================================
# 10. エントリポイント
# ==============================================================================
function main(; kwargs...)
    conf = Config(; kwargs...)

    df, seed_df = compute_background(conf)

    csv_path  = joinpath(PLOT_DIR, "$(conf.file_stub)_section_points.csv")
    seed_path = joinpath(PLOT_DIR, "$(conf.file_stub)_seed_points.csv")
    png_path  = joinpath(PLOT_DIR, "$(conf.file_stub)_section_xvx.png")

    if conf.save_csv
        CSV.write(csv_path,  df)
        CSV.write(seed_path, seed_df)
    end

    fig = plot_background(df, seed_df, conf; plot_seeds=false)

    conf.save_png && save(png_path, fig; px_per_unit=2)

    println("\n=== 保存先 ===")
    conf.save_csv && println(csv_path)
    conf.save_csv && println(seed_path)
    conf.save_png && println(png_path)

    if nrow(df) > 0
        @printf("\nC 誤差  min=%.3e  max=%.3e\n",
                minimum(df.C_error), maximum(df.C_error))
        @printf("x  範囲 [%.6f, %.6f]\n", minimum(df.x),  maximum(df.x))
        @printf("ẋ  範囲 [%.6f, %.6f]\n", minimum(df.vx), maximum(df.vx))
    else
        println("交差点が収集されませんでした。seed 範囲を確認してください。")
    end

    return df, seed_df, fig
end

end  # module GanimedePoincareBackground

# ==============================================================================
# Run
# ==============================================================================
using .GanimedePoincareBackground

# ----------------------------------------------------------------
# 「少数 seed × 長時間積分」設定
#
# ポイント:
#   nx=20, nvx=20 → 合計 ~400 seed (許容域のみ有効)
#   Smax=1200, tmax=8000 → 各軌道を長く追い、
#     KAM トーラス → 閉じた点列
#     共鳴島 (island chain) → 孤立点群
#     カオス海 → 広い散在点群
#   が鮮明に現れる。
#
# 計算時間の目安: ~数分 (CPU, シングルスレッド)
# ----------------------------------------------------------------
GanimedePoincareBackground.main(
    nx  = 160,
    nvx = 80,

    x_min_seed   = -0.99,
    x_max_seed   = -0.7,
    vx_min_seed  = -0.05,
    vx_max_seed  =  0.05,

    initial_vy_branch = :pos,
    vy_filter         = :none,   # 両方向の交差を収集

    Smax  = 1200,
    tmax  = 8000.0,

    reltol = 1e-12,
    abstol = 1e-12,

    xlim = (-1.20, -0.50),
    ylim = (-0.35,  0.35),

    markersize = 3.5,
)
