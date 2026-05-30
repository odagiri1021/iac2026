module GanimedePoincareManifolds

using DifferentialEquations
using LinearAlgebra
using GLMakie
using Printf
using Colors
using FileIO

# ==============================================================================
# 0. Settings (Jupiter–Ganymede planar CR3BP)
# ==============================================================================
const MU = 7.806548101361265e-05

const ROW = (
    x0     = 0.958000406,
    y0     = 0.0,
    vx0    = 0.0,
    vy0    = -0.045142641,
    T      = 31.58106166,
    order  = 3,
    p      = 3,
    q      = 5,
    q_err  = 0.026282071,
    stable = false,
    rho    = 51.81466564
)


const BASE_DIR = @__DIR__
const PLOT_DIR = joinpath(BASE_DIR, "plots_ganimede_manifold")

mkpath(PLOT_DIR)

# ==============================================================================
# 1. Params
# ==============================================================================
struct CR3BPParams
    μ::Float64
end
const P = CR3BPParams(MU)

# ==============================================================================
# 2. Planar CR3BP dynamics: u = [x, y, vx, vy]
# ==============================================================================
function cr3bp_dynamics!(du, u, p::CR3BPParams, t)
    @inbounds begin
        μ  = p.μ
        x  = u[1];  y  = u[2]
        vx = u[3];  vy = u[4]

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
# 3. CR3BP + STM: u = [x,y,vx,vy, vec(Phi)]  (Phi 4x4 col-major)
# ==============================================================================
const PHI0_VEC = let M = Matrix{Float64}(I, 4, 4); vec(M) end

function cr3bp_stm_dynamics!(du, u, p::CR3BPParams, t)
    @inbounds begin
        μ  = p.μ
        x  = u[1];  y  = u[2]
        vx = u[3];  vy = u[4]

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

        # Jacobian terms
        Ωxx = 1 - om*inv_r1_3 - μ*inv_r2_3 + 3*om*(x+μ)*(x+μ)*inv_r1_5 + 3*μ*(x-1+μ)*(x-1+μ)*inv_r2_5
        Ωxy = 3*om*(x+μ)*y*inv_r1_5 + 3*μ*(x-1+μ)*y*inv_r2_5
        Ωyy = 1 - om*inv_r1_3 - μ*inv_r2_3 + 3*om*y*y*inv_r1_5 + 3*μ*y*y*inv_r2_5

        # STM
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
# 4. Abort callback
# ==============================================================================
function make_abort_callback(; limit=10.0)
    condition(u, t, integrator) = (abs(u[1]) > limit) || (abs(u[2]) > limit)
    affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

# ==============================================================================
# 5. Integrate one period with STM to obtain monodromy Φ(T)
# ==============================================================================
function integrate_orbit_with_stm(u0_state::NTuple{4,Float64}, T::Float64, p::CR3BPParams;
                                  reltol::Float64=3e-14, abstol::Float64=1e-14)
    u0 = zeros(20)
    u0[1] = u0_state[1]
    u0[2] = u0_state[2]
    u0[3] = u0_state[3]
    u0[4] = u0_state[4]
    @inbounds for i in 1:16
        u0[4+i] = PHI0_VEC[i]
    end

    prob = ODEProblem(cr3bp_stm_dynamics!, u0, (0.0, T), p)
    sol  = solve(prob, Vern9();
                 callback=make_abort_callback(limit=10.0),
                 reltol=reltol, abstol=abstol,
                 save_everystep=false)

    xf = sol.u[end]
    ΦT = reshape(@view(xf[5:20]), 4, 4)
    return ΦT
end

# ==============================================================================
# 6. Stable/unstable eigenvectors from monodromy
# ==============================================================================
function stable_unstable_eigvecs(ΦT::AbstractMatrix{<:Real})
    Φ = Matrix{Float64}(ΦT)
    ev = eigen(Φ)
    λ  = ev.values
    V  = ev.vectors

    idx = sortperm(abs.(λ .- 1.0))
    trivial = idx[1:2]
    others = [k for k in 1:4 if !(k in trivial)]

    absλ = abs.(λ[others])
    iu = others[argmax(absλ)]
    is = others[argmin(absλ)]

    wu = real.(V[:, iu])
    ws = real.(V[:, is])

    return λ[iu], wu, λ[is], ws
end

@inline function normalize_by_position(v::Vector{Float64})
    npos = norm(@view(v[1:2]))
    if npos > 0
        return v ./ npos
    end
    n = norm(v)
    return (n > 0) ? (v ./ n) : v
end

# ==============================================================================
# 7. Collect crossings on Σ: y=0, with filters
#    s counts ONLY crossings that satisfy x<0 (and vy_filter if set).
# ==============================================================================
function collect_poincare_hits(u0_state::Vector{Float64}, p::CR3BPParams;
                               tspan::Tuple{Float64,Float64},
                               Smax::Int=30,
                               xneg::Bool=true,
                               vy_filter::Symbol=:none,   # :none / :pos / :neg
                               reltol::Float64=1e-12,
                               abstol::Float64=1e-12,
                               abort_limit::Float64=10.0,
                               t_ignore::Float64=1e-10,
                               dt_min::Float64=1e-8)

    xs  = Float64[]
    vxs = Float64[]
    last_t = Ref(NaN)

    condition(u, t, integrator) = u[2]  # y = 0

    function affect!(integrator)
        t = integrator.t
        if abs(t) < t_ignore
            return
        end
        if !isnan(last_t[]) && abs(t - last_t[]) < dt_min
            return
        end
        last_t[] = t

        x  = integrator.u[1]
        vx = integrator.u[3]
        vy = integrator.u[4]

        if xneg && !(x < 0.0)
            return
        end

        if vy_filter === :pos
            if !(vy > 0.0); return; end
        elseif vy_filter === :neg
            if !(vy < 0.0); return; end
        end

        push!(xs, x)
        push!(vxs, vx)

        if length(xs) >= Smax
            terminate!(integrator)
        end
    end

    cb_p = ContinuousCallback(condition, affect!; save_positions=(false,false))
    cb_abort = DiscreteCallback(
        (u,t,integrator) -> (abs(u[1]) > abort_limit) || (abs(u[2]) > abort_limit),
        integrator -> terminate!(integrator);
        save_positions=(false,false)
    )

    prob = ODEProblem(cr3bp_dynamics!, u0_state, tspan, p)
    solve(prob, Vern9();
          callback=CallbackSet(cb_p, cb_abort),
          reltol=reltol, abstol=abstol,
          save_everystep=false)

    return xs, vxs
end

# ==============================================================================
# Result container
# ==============================================================================
struct PoincareBranch
    Xp::Matrix{Float64}
    Vp::Matrix{Float64}
    Xm::Matrix{Float64}
    Vm::Matrix{Float64}
end

struct PoincareResult
    Wu::PoincareBranch
    Ws::PoincareBranch
    n::Int
    eps::Float64
    Smax::Int
    tmax::Float64
    vy_filter::Symbol
    λu
    λs
    wu::Vector{Float64}
    ws::Vector{Float64}
end

const LAST_RESULT = Ref{Union{Nothing,PoincareResult}}(nothing)

# ==============================================================================
# 8A. Compute only
# ==============================================================================
function compute(u0_state::NTuple{4,Float64}=(ROW.x0, ROW.y0, ROW.vx0, ROW.vy0),
                 T::Float64=ROW.T,
                 p::CR3BPParams=P;
                 n::Int=1000,
                 eps::Float64=1e-11,
                 Smax::Int=10,
                 tmax::Float64=400.0,
                 abort_limit::Float64=10.0,
                 vy_filter::Symbol=:none)

    ΦT = integrate_orbit_with_stm(u0_state, T, p)
    λu, wu0, λs, ws0 = stable_unstable_eigvecs(ΦT)
    wu0 = normalize_by_position(Vector(wu0))
    ws0 = normalize_by_position(Vector(ws0))

    @printf("monodromy: |λu|=%.6e  |λs|=%.6e\n", abs(λu), abs(λs))

    function compute_hits_for_vector(w::Vector{Float64}, tspan::Tuple{Float64,Float64})
        Xp = fill(NaN, Smax, n)
        Vp = fill(NaN, Smax, n)
        Xm = fill(NaN, Smax, n)
        Vm = fill(NaN, Smax, n)

        u0 = collect(u0_state)

        for k in 1:n
            δ = k * eps

            # + branch
            u_plus = u0 .+ δ .* w
            xs, vxs = collect_poincare_hits(Vector(u_plus), p;
                                            tspan=tspan, Smax=Smax,
                                            xneg=true, vy_filter=vy_filter,
                                            abort_limit=abort_limit)
            for s in 1:min(Smax, length(xs))
                Xp[s,k] = xs[s]
                Vp[s,k] = vxs[s]
            end

            # - branch
            u_minus = u0 .- δ .* w
            xs2, vxs2 = collect_poincare_hits(Vector(u_minus), p;
                                              tspan=tspan, Smax=Smax,
                                              xneg=true, vy_filter=vy_filter,
                                              abort_limit=abort_limit)
            for s in 1:min(Smax, length(xs2))
                Xm[s,k] = xs2[s]
                Vm[s,k] = vxs2[s]
            end
        end

        return PoincareBranch(Xp, Vp, Xm, Vm)
    end

    Wu = compute_hits_for_vector(wu0, (0.0,  tmax))   # unstable forward
    Ws = compute_hits_for_vector(ws0, (0.0, -tmax))   # stable backward

    res = PoincareResult(Wu, Ws, n, eps, Smax, tmax, vy_filter, λu, λs, wu0, ws0)
    LAST_RESULT[] = res
    return res
end

# main = compute (名前を残す)
main(; kwargs...) = compute(; kwargs...)

# ==============================================================================
# 8B. Plot only
# ==============================================================================
# ==============================================================================
# 8B. Plot only
# ==============================================================================
# ==============================================================================
# 8B. Plot only
# ==============================================================================
function plot(res::PoincareResult=let r=LAST_RESULT[]; r===nothing ? error("Run main() or compute() first.") : r end;
              s_u_only::Union{Nothing,Int}=nothing, # 不安定方向(赤)の交差回数
              s_s_only::Union{Nothing,Int}=nothing, # 安定方向(青)の交差回数
              connect::Bool=true,
              scatter_all::Bool=true,
              xlim::Tuple{Float64,Float64}=(-1.2, -0.5),
              ylim::Tuple{Float64,Float64}=(-0.2, 0.2),
              markersize::Float64=2.8,
              linewidth::Float64=1.2,
              gamma::Float64=3.0)

    n = res.n
    Smax = res.Smax

    # ----- 入力チェック -----
    if !(isnothing(s_u_only)) && (s_u_only < 1 || s_u_only > Smax)
        error("s_u_only must satisfy 1 <= s_u_only <= Smax. Got $s_u_only")
    end
    if !(isnothing(s_s_only)) && (s_s_only < 1 || s_s_only > Smax)
        error("s_s_only must satisfy 1 <= s_s_only <= Smax. Got $s_s_only")
    end

    # ----- color helpers -----
    lerp(c0::RGB, c1::RGB, t) = RGB(
        (1-t)*red(c0)   + t*red(c1),
        (1-t)*green(c0) + t*green(c1),
        (1-t)*blue(c0)  + t*blue(c1),
    )

    t_of_k(k, n) = (n <= 1) ? 0.0 : ((k-1) / (n-1))^gamma

    base_red    = RGB(1.0, 0.0, 0.0)
    pink_far    = RGB(0.0, 1.0, 0.0)
    darkred_far = RGB(0.05, 0.0, 0.0)

    base_blue     = RGB(0.0, 0.0, 1.0)
    lightblue_far = RGB(0.0, 1.0, 0.0)
    darkblue_far  = RGB(0.0, 0.0, 0.05)

    col_u_plus(k,n)  = lerp(base_red,  darkred_far,   t_of_k(k,n))
    col_u_minus(k,n) = lerp(base_red,  pink_far,      t_of_k(k,n))
    col_s_plus(k,n)  = lerp(base_blue, darkblue_far,  t_of_k(k,n))
    col_s_minus(k,n) = lerp(base_blue, lightblue_far, t_of_k(k,n))

    # ==========================================================
    # 1. x-vx 平面 (ポアンカレ切断面) の作成
    # ==========================================================
    fig_xvx = Figure(size=(1200, 900))
    str_su = isnothing(s_u_only) ? "all" : string(s_u_only)
    str_ss = isnothing(s_s_only) ? "all" : string(s_s_only)
    
    title_pc = @sprintf("Σ: y=0, x<0 (x,vx)  p:q=%d:%d  n=%d eps=%.1e Smax=%d vy=%s  s_u=%s, s_s=%s  gamma=%.2f",
                        ROW.p, ROW.q, res.n, res.eps, res.Smax, String(res.vy_filter),
                        str_su, str_ss, gamma)
    ax_xvx = Axis(fig_xvx[1,1], title=title_pc)

    function scatter_from_matrix!(ax, X::Matrix{Float64}, V::Matrix{Float64}, colfun;
                                  s_only::Union{Nothing,Int}=nothing, ms=2.8)
        xs = Float32[]
        vs = Float32[]
        cs = RGB{Float64}[]
        S, K = size(X)
        srange = isnothing(s_only) ? (1:S) : (s_only:s_only)
        for s in srange
            for k in 1:K
                x = X[s,k]; v = V[s,k]
                if isfinite(x) && isfinite(v)
                    push!(xs, Float32(x))
                    push!(vs, Float32(v))
                    push!(cs, colfun(k, K))
                end
            end
        end
        if !isempty(xs)
            scatter!(ax, xs, vs, markersize=ms, color=cs)
        end
        return nothing
    end

    function connect_same_s!(ax, X::Matrix{Float64}, V::Matrix{Float64},
                             srange, col; lw=1.2)
        S, K = size(X)
        for s in srange
            segx = Float32[]
            segv = Float32[]
            for k in 1:K
                x = X[s,k]; v = V[s,k]
                if isfinite(x) && isfinite(v)
                    push!(segx, Float32(x))
                    push!(segv, Float32(v))
                else
                    if length(segx) >= 2
                        lines!(ax, segx, segv, linewidth=lw, color=col)
                    end
                    empty!(segx); empty!(segv)
                end
            end
            if length(segx) >= 2
                lines!(ax, segx, segv, linewidth=lw, color=col)
            end
        end
        return nothing
    end

    srange_u = isnothing(s_u_only) ? (1:Smax) : (s_u_only:s_u_only)
    srange_s = isnothing(s_s_only) ? (1:Smax) : (s_s_only:s_s_only)

    if scatter_all
        scatter_from_matrix!(ax_xvx, res.Wu.Xp, res.Wu.Vp, col_u_plus;  s_only=s_u_only, ms=markersize)
        scatter_from_matrix!(ax_xvx, res.Wu.Xm, res.Wu.Vm, col_u_minus; s_only=s_u_only, ms=markersize)
        scatter_from_matrix!(ax_xvx, res.Ws.Xp, res.Ws.Vp, col_s_plus;  s_only=s_s_only, ms=markersize)
        scatter_from_matrix!(ax_xvx, res.Ws.Xm, res.Ws.Vm, col_s_minus; s_only=s_s_only, ms=markersize)
    end

    if connect
        connect_same_s!(ax_xvx, res.Wu.Xp, res.Wu.Vp, srange_u, :red;  lw=linewidth)
        connect_same_s!(ax_xvx, res.Wu.Xm, res.Wu.Vm, srange_u, :red;  lw=linewidth)
        connect_same_s!(ax_xvx, res.Ws.Xp, res.Ws.Vp, srange_s, :blue; lw=linewidth)
        connect_same_s!(ax_xvx, res.Ws.Xm, res.Ws.Vm, srange_s, :blue; lw=linewidth)
    end

    # 周期解の点 (k=1, eps=0) を黒でプロット
    xu_base = Float32[res.Wu.Xp[s, 1] for s in srange_u if isfinite(res.Wu.Xp[s, 1])]
    vu_base = Float32[res.Wu.Vp[s, 1] for s in srange_u if isfinite(res.Wu.Vp[s, 1])]
    if !isempty(xu_base)
        scatter!(ax_xvx, xu_base, vu_base, color=:black, markersize=markersize*3.0)
    end

    xs_base = Float32[res.Ws.Xp[s, 1] for s in srange_s if isfinite(res.Ws.Xp[s, 1])]
    vs_base = Float32[res.Ws.Vp[s, 1] for s in srange_s if isfinite(res.Ws.Vp[s, 1])]
    if !isempty(xs_base)
        scatter!(ax_xvx, xs_base, vs_base, color=:black, markersize=markersize*3.0)
    end

    xlims!(ax_xvx, -1.1, -0.5)
    ylims!(ax_xvx, -0.25, 0.25)

    # ==========================================================
    # 2. x-y 平面 (周期解ベース軌道) の作成
    # ==========================================================
    fig_xy = Figure(size=(900, 900))
    title_xy = @sprintf("Periodic Orbit in x-y plane  p:q=%d:%d", ROW.p, ROW.q)
    
    # 軌道の形を崩さないように aspect=DataAspect() で縦横比を1:1に固定
    ax_xy = Axis(fig_xy[1,1], title=title_xy, aspect=DataAspect())

    # 1周期分の軌道を積分
    u0_base = [ROW.x0, ROW.y0, ROW.vx0, ROW.vy0]
    prob_xy = ODEProblem(cr3bp_dynamics!, u0_base, (0.0, ROW.T), P)
    sol_xy  = solve(prob_xy, Vern9(), reltol=1e-12, abstol=1e-12, save_everystep=true)

    # 軌道の描画
    lines!(ax_xy, sol_xy[1,:], sol_xy[2,:], color=:red, linewidth=1.5)

    # 木星(-MU, 0) と ガニメデ(1-MU, 0) を青い点で描画
    scatter!(ax_xy, [-MU, 1.0 - MU], [0.0, 0.0], color=:blue, markersize=8.0)

    # 戻り値を2つのFigureにする
    return fig_xy, fig_xvx
end

# convenience save
function savefig(fig, filename::AbstractString)
    FileIO.save(filename, fig)
    return filename
end

end # module

using .GanimedePoincareManifolds

res = GanimedePoincareManifolds.main(n=3000
, eps=3e-5, Smax=5, tmax=200.0, vy_filter=:none)

# 戻り値を2つ受け取る
fig_xy, fig_xvx = GanimedePoincareManifolds.plot(res; s_s_only=2, s_u_only=2, connect=false, scatter_all=true, gamma=1.0)

# それぞれ保存する
GanimedePoincareManifolds.savefig(fig_xy,  joinpath(GanimedePoincareManifolds.PLOT_DIR, "orbit_xy.png"))
GanimedePoincareManifolds.savefig(fig_xvx, joinpath(GanimedePoincareManifolds.PLOT_DIR, "manifold_xvx.png"))

# Notebook環境などで両方表示させたい場合は、個別に評価して表示させます
# display(fig_xy)
# display(fig_xvx)