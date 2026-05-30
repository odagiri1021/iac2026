module GanimedePoincareManifolds

using DifferentialEquations
using LinearAlgebra
using GLMakie
using Printf
using Colors
using FileIO

using CairoMakie
CairoMakie.activate!()

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
        x  = u[1]
        y  = u[2]
        vx = u[3]
        vy = u[4]

        r1sq = (x + μ) * (x + μ) + y * y
        r2sq = (x - 1 + μ) * (x - 1 + μ) + y * y

        r1 = sqrt(r1sq)
        r2 = sqrt(r2sq)

        inv_r1_3 = 1.0 / (r1sq * r1)
        inv_r2_3 = 1.0 / (r2sq * r2)

        om = 1.0 - μ

        du[1] = vx
        du[2] = vy
        du[3] = 2 * vy + x - om * (x + μ) * inv_r1_3 - μ * (x - 1 + μ) * inv_r2_3
        du[4] = -2 * vx + y - om * y * inv_r1_3 - μ * y * inv_r2_3
    end

    return nothing
end

# ==============================================================================
# 3. CR3BP + STM: u = [x, y, vx, vy, vec(Phi)]
# ==============================================================================
const PHI0_VEC = let M = Matrix{Float64}(I, 4, 4)
    vec(M)
end

function cr3bp_stm_dynamics!(du, u, p::CR3BPParams, t)
    @inbounds begin
        μ  = p.μ
        x  = u[1]
        y  = u[2]
        vx = u[3]
        vy = u[4]

        r1sq = (x + μ) * (x + μ) + y * y
        r2sq = (x - 1 + μ) * (x - 1 + μ) + y * y

        r1 = sqrt(r1sq)
        r2 = sqrt(r2sq)

        inv_r1_3 = 1.0 / (r1sq * r1)
        inv_r2_3 = 1.0 / (r2sq * r2)
        inv_r1_5 = inv_r1_3 / r1sq
        inv_r2_5 = inv_r2_3 / r2sq

        om = 1.0 - μ

        du[1] = vx
        du[2] = vy
        du[3] = 2 * vy + x - om * (x + μ) * inv_r1_3 - μ * (x - 1 + μ) * inv_r2_3
        du[4] = -2 * vx + y - om * y * inv_r1_3 - μ * y * inv_r2_3

        Ωxx = 1 - om * inv_r1_3 - μ * inv_r2_3 +
              3 * om * (x + μ) * (x + μ) * inv_r1_5 +
              3 * μ * (x - 1 + μ) * (x - 1 + μ) * inv_r2_5

        Ωxy = 3 * om * (x + μ) * y * inv_r1_5 +
              3 * μ * (x - 1 + μ) * y * inv_r2_5

        Ωyy = 1 - om * inv_r1_3 - μ * inv_r2_3 +
              3 * om * y * y * inv_r1_5 +
              3 * μ * y * y * inv_r2_5

        base = 5

        for j in 0:3
            k = base + 4 * j

            φ1 = u[k]
            φ2 = u[k + 1]
            φ3 = u[k + 2]
            φ4 = u[k + 3]

            du[k]     = φ3
            du[k + 1] = φ4
            du[k + 2] = Ωxx * φ1 + Ωxy * φ2 + 2 * φ4
            du[k + 3] = Ωxy * φ1 + Ωyy * φ2 - 2 * φ3
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
function integrate_orbit_with_stm(u0_state::NTuple{4,Float64},
                                  T::Float64,
                                  p::CR3BPParams;
                                  reltol::Float64=3e-14,
                                  abstol::Float64=1e-14)

    u0 = zeros(20)

    u0[1] = u0_state[1]
    u0[2] = u0_state[2]
    u0[3] = u0_state[3]
    u0[4] = u0_state[4]

    @inbounds for i in 1:16
        u0[4 + i] = PHI0_VEC[i]
    end

    prob = ODEProblem(cr3bp_stm_dynamics!, u0, (0.0, T), p)

    sol = solve(prob, Vern9();
                callback=make_abort_callback(limit=10.0),
                reltol=reltol,
                abstol=abstol,
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
    λ = ev.values
    V = ev.vectors

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

    if npos > 0.0
        return v ./ npos
    end

    n = norm(v)

    return n > 0.0 ? v ./ n : v
end

# ==============================================================================
# 7. Collect crossings on Σ: y=0
# ==============================================================================
function collect_poincare_hits(u0_state::Vector{Float64},
                               p::CR3BPParams;
                               tspan::Tuple{Float64,Float64},
                               Smax::Int=30,
                               xneg::Bool=true,
                               vy_filter::Symbol=:none,
                               reltol::Float64=1e-12,
                               abstol::Float64=1e-12,
                               abort_limit::Float64=10.0,
                               t_ignore::Float64=1e-10,
                               dt_min::Float64=1e-8)

    xs = Float64[]
    vxs = Float64[]
    last_t = Ref(NaN)

    condition(u, t, integrator) = u[2]

    function affect!(integrator)
        t = integrator.t

        if abs(t) < t_ignore
            return
        end

        if !isnan(last_t[]) && abs(t - last_t[]) < dt_min
            return
        end

        last_t[] = t

        x = integrator.u[1]
        vx = integrator.u[3]
        vy = integrator.u[4]

        if xneg && !(x < 0.0)
            return
        end

        if vy_filter === :pos
            if !(vy > 0.0)
                return
            end
        elseif vy_filter === :neg
            if !(vy < 0.0)
                return
            end
        end

        push!(xs, x)
        push!(vxs, vx)

        if length(xs) >= Smax
            terminate!(integrator)
        end

        return
    end

    cb_p = ContinuousCallback(condition, affect!; save_positions=(false, false))

    cb_abort = DiscreteCallback(
        (u, t, integrator) -> (abs(u[1]) > abort_limit) || (abs(u[2]) > abort_limit),
        integrator -> terminate!(integrator);
        save_positions=(false, false)
    )

    prob = ODEProblem(cr3bp_dynamics!, u0_state, tspan, p)

    solve(prob, Vern9();
          callback=CallbackSet(cb_p, cb_abort),
          reltol=reltol,
          abstol=abstol,
          save_everystep=false)

    return xs, vxs
end

# ==============================================================================
# 8. Result container
# ==============================================================================
struct PoincareBranch
    Xp::Matrix{Float64}
    Vp::Matrix{Float64}
    δp::Vector{Float64}

    Xm::Matrix{Float64}
    Vm::Matrix{Float64}
    δm::Vector{Float64}
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

    refine_distance::Float64
    refine_s::Union{Symbol,Int}
    max_refine_iter::Int
end

const LAST_RESULT = Ref{Union{Nothing,PoincareResult}}(nothing)

# ==============================================================================
# 9. Compute only
# ==============================================================================
function compute(u0_state::NTuple{4,Float64}=(ROW.x0, ROW.y0, ROW.vx0, ROW.vy0),
                 T::Float64=ROW.T,
                 p::CR3BPParams=P;
                 n::Int=1000,
                 eps::Float64=1e-11,
                 Smax::Int=10,
                 tmax::Float64=400.0,
                 abort_limit::Float64=10.0,
                 vy_filter::Symbol=:none,
                 refine_distance::Float64=0.01,
                 refine_s::Union{Symbol,Int}=:all,
                 max_refine_iter::Int=10,
                 max_points_per_branch::Int=30000)

    if refine_s isa Int
        if refine_s < 1 || refine_s > Smax
            error("refine_s must satisfy 1 <= refine_s <= Smax. Got $refine_s")
        end
    elseif refine_s !== :all
        error("refine_s must be :all or an integer.")
    end

    ΦT = integrate_orbit_with_stm(u0_state, T, p)

    λu, wu0, λs, ws0 = stable_unstable_eigvecs(ΦT)

    wu0 = normalize_by_position(Vector(wu0))
    ws0 = normalize_by_position(Vector(ws0))

    @printf("monodromy: |λu|=%.6e  |λs|=%.6e\n", abs(λu), abs(λs))

    u_base = collect(u0_state)

    function propagate_delta(w::Vector{Float64},
                             branchsign::Float64,
                             δ::Float64,
                             tspan::Tuple{Float64,Float64})

        u_init = u_base .+ (branchsign * δ) .* w

        xs, vxs = collect_poincare_hits(Vector(u_init), p;
                                        tspan=tspan,
                                        Smax=Smax,
                                        xneg=true,
                                        vy_filter=vy_filter,
                                        abort_limit=abort_limit)

        Xcol = fill(NaN, Smax)
        Vcol = fill(NaN, Smax)

        for s in 1:min(Smax, length(xs))
            Xcol[s] = xs[s]
            Vcol[s] = vxs[s]
        end

        return Xcol, Vcol
    end

    function section_distance(Xa::Vector{Float64},
                              Va::Vector{Float64},
                              Xb::Vector{Float64},
                              Vb::Vector{Float64})

        if refine_s === :all
            dmax = 0.0

            for s in 1:Smax
                if isfinite(Xa[s]) && isfinite(Va[s]) &&
                   isfinite(Xb[s]) && isfinite(Vb[s])

                    d = hypot(Xa[s] - Xb[s], Va[s] - Vb[s])
                    dmax = max(dmax, d)
                end
            end

            return dmax
        else
            s = refine_s

            if isfinite(Xa[s]) && isfinite(Va[s]) &&
               isfinite(Xb[s]) && isfinite(Vb[s])

                return hypot(Xa[s] - Xb[s], Va[s] - Vb[s])
            else
                return 0.0
            end
        end
    end

    function cols_to_matrix(cols::Vector{Vector{Float64}})
        M = fill(NaN, Smax, length(cols))

        for k in eachindex(cols)
            M[:, k] .= cols[k]
        end

        return M
    end

    function compute_adaptive_branch(w::Vector{Float64},
                                     branchsign::Float64,
                                     tspan::Tuple{Float64,Float64},
                                     label::String)

        δs = [k * eps for k in 1:n]

        Xcols = Vector{Vector{Float64}}(undef, length(δs))
        Vcols = Vector{Vector{Float64}}(undef, length(δs))

        for k in eachindex(δs)
            Xcols[k], Vcols[k] = propagate_delta(w, branchsign, δs[k], tspan)
        end

        initial_N = length(δs)

        for iter in 1:max_refine_iter
            inserted = false

            newδs = Float64[]
            newXcols = Vector{Vector{Float64}}()
            newVcols = Vector{Vector{Float64}}()

            sizehint!(newδs, min(2 * length(δs), max_points_per_branch))
            sizehint!(newXcols, min(2 * length(δs), max_points_per_branch))
            sizehint!(newVcols, min(2 * length(δs), max_points_per_branch))

            push!(newδs, δs[1])
            push!(newXcols, Xcols[1])
            push!(newVcols, Vcols[1])

            for k in 1:(length(δs) - 1)
                d = section_distance(Xcols[k], Vcols[k], Xcols[k + 1], Vcols[k + 1])

                if d >= refine_distance
                    if length(newδs) + 1 < max_points_per_branch
                        δmid = 0.5 * (δs[k] + δs[k + 1])

                        Xmid, Vmid = propagate_delta(w, branchsign, δmid, tspan)

                        push!(newδs, δmid)
                        push!(newXcols, Xmid)
                        push!(newVcols, Vmid)

                        inserted = true
                    end
                end

                push!(newδs, δs[k + 1])
                push!(newXcols, Xcols[k + 1])
                push!(newVcols, Vcols[k + 1])

                if length(newδs) >= max_points_per_branch
                    break
                end
            end

            δs = newδs
            Xcols = newXcols
            Vcols = newVcols

            @printf("%s refine iter %d / %d: N = %d\n",
                    label, iter, max_refine_iter, length(δs))

            if !inserted
                @printf("%s refinement converged before reaching %d iterations.\n",
                        label, max_refine_iter)
                break
            end

            if length(δs) >= max_points_per_branch
                @printf("%s reached max_points_per_branch = %d\n",
                        label, max_points_per_branch)
                break
            end
        end

        @printf("%s final: %d -> %d points\n", label, initial_N, length(δs))

        X = cols_to_matrix(Xcols)
        V = cols_to_matrix(Vcols)

        return X, V, δs
    end

    function compute_hits_for_vector(w::Vector{Float64},
                                     tspan::Tuple{Float64,Float64},
                                     label::String)

        Xp, Vp, δp = compute_adaptive_branch(w, +1.0, tspan, label * " +")
        Xm, Vm, δm = compute_adaptive_branch(w, -1.0, tspan, label * " -")

        return PoincareBranch(Xp, Vp, δp, Xm, Vm, δm)
    end

    Wu = compute_hits_for_vector(wu0, (0.0,  tmax), "Wu")
    Ws = compute_hits_for_vector(ws0, (0.0, -tmax), "Ws")

    n_actual = maximum((
        length(Wu.δp),
        length(Wu.δm),
        length(Ws.δp),
        length(Ws.δm),
    ))

    res = PoincareResult(Wu, Ws,
                         n_actual,
                         eps,
                         Smax,
                         tmax,
                         vy_filter,
                         λu,
                         λs,
                         wu0,
                         ws0,
                         refine_distance,
                         refine_s,
                         max_refine_iter)

    LAST_RESULT[] = res

    return res
end

main(; kwargs...) = compute(; kwargs...)

# ==============================================================================
# 10. Plot only
# ==============================================================================
function plot(res::PoincareResult=let r = LAST_RESULT[]
                  r === nothing ? error("Run main() or compute() first.") : r
              end;
              s_u_only::Union{Nothing,Int}=nothing,
              s_s_only::Union{Nothing,Int}=nothing,
              connect::Bool=true,
              scatter_all::Bool=true,
              xlim::Tuple{Float64,Float64}=(-1.2, -0.5),
              ylim::Tuple{Float64,Float64}=(-0.2, 0.2),
              markersize::Float64=2.8,
              linewidth::Float64=1.2,
              gamma::Float64=3.0)

    Smax = res.Smax

    if !(isnothing(s_u_only)) && (s_u_only < 1 || s_u_only > Smax)
        error("s_u_only must satisfy 1 <= s_u_only <= Smax. Got $s_u_only")
    end

    if !(isnothing(s_s_only)) && (s_s_only < 1 || s_s_only > Smax)
        error("s_s_only must satisfy 1 <= s_s_only <= Smax. Got $s_s_only")
    end

    lerp(c0::RGB, c1::RGB, t) = RGB(
        (1 - t) * red(c0)   + t * red(c1),
        (1 - t) * green(c0) + t * green(c1),
        (1 - t) * blue(c0)  + t * blue(c1),
    )

    t_of_k(k, n) = n <= 1 ? 0.0 : ((k - 1) / (n - 1))^gamma

    base_red = RGB(1.0, 0.0, 0.0)
    green_far_for_u_minus = RGB(0.0, 1.0, 0.0)
    darkred_far = RGB(0.05, 0.0, 0.0)

    base_blue = RGB(0.0, 0.0, 1.0)
    green_far_for_s_minus = RGB(0.0, 1.0, 0.0)
    darkblue_far = RGB(0.0, 0.0, 0.05)

    col_u_plus(k, n)  = lerp(base_red,  darkred_far,          t_of_k(k, n))
    col_u_minus(k, n) = lerp(base_red,  green_far_for_u_minus, t_of_k(k, n))
    col_s_plus(k, n)  = lerp(base_blue, darkblue_far,         t_of_k(k, n))
    col_s_minus(k, n) = lerp(base_blue, green_far_for_s_minus, t_of_k(k, n))

    fig_xvx = Figure(size=(1200, 900))

    str_su = isnothing(s_u_only) ? "all" : string(s_u_only)
    str_ss = isnothing(s_s_only) ? "all" : string(s_s_only)
    str_rs = res.refine_s === :all ? "all" : string(res.refine_s)

    title_pc = @sprintf(
        "Σ: y=0, x<0 (x,vx)  p:q=%d:%d  Nmax=%d eps=%.1e Smax=%d vy=%s  refine_dist=%.3g refine_s=%s iter=%d  s_u=%s s_s=%s",
        ROW.p,
        ROW.q,
        res.n,
        res.eps,
        res.Smax,
        String(res.vy_filter),
        res.refine_distance,
        str_rs,
        res.max_refine_iter,
        str_su,
        str_ss,
    )

    ax_xvx = Axis(fig_xvx[1, 1], title=title_pc)

    function scatter_from_matrix!(ax,
                                  X::Matrix{Float64},
                                  V::Matrix{Float64},
                                  colfun;
                                  s_only::Union{Nothing,Int}=nothing,
                                  ms=2.8)

        xs = Float32[]
        vs = Float32[]
        cs = RGB{Float64}[]

        S, K = size(X)

        srange = isnothing(s_only) ? (1:S) : (s_only:s_only)

        for s in srange
            for k in 1:K
                x = X[s, k]
                v = V[s, k]

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

    function connect_same_s!(ax,
                             X::Matrix{Float64},
                             V::Matrix{Float64},
                             srange,
                             col;
                             lw=1.2)

        S, K = size(X)

        for s in srange
            segx = Float32[]
            segv = Float32[]

            for k in 1:K
                x = X[s, k]
                v = V[s, k]

                if isfinite(x) && isfinite(v)
                    push!(segx, Float32(x))
                    push!(segv, Float32(v))
                else
                    if length(segx) >= 2
                        lines!(ax, segx, segv, linewidth=lw, color=col)
                    end

                    empty!(segx)
                    empty!(segv)
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
        scatter_from_matrix!(ax_xvx, res.Wu.Xp, res.Wu.Vp, col_u_plus;
                             s_only=s_u_only, ms=markersize)

        scatter_from_matrix!(ax_xvx, res.Wu.Xm, res.Wu.Vm, col_u_minus;
                             s_only=s_u_only, ms=markersize)

        scatter_from_matrix!(ax_xvx, res.Ws.Xp, res.Ws.Vp, col_s_plus;
                             s_only=s_s_only, ms=markersize)

        scatter_from_matrix!(ax_xvx, res.Ws.Xm, res.Ws.Vm, col_s_minus;
                             s_only=s_s_only, ms=markersize)
    end

    if connect
        connect_same_s!(ax_xvx, res.Wu.Xp, res.Wu.Vp, srange_u, :red;
                        lw=linewidth)

        connect_same_s!(ax_xvx, res.Wu.Xm, res.Wu.Vm, srange_u, :red;
                        lw=linewidth)

        connect_same_s!(ax_xvx, res.Ws.Xp, res.Ws.Vp, srange_s, :blue;
                        lw=linewidth)

        connect_same_s!(ax_xvx, res.Ws.Xm, res.Ws.Vm, srange_s, :blue;
                        lw=linewidth)
    end

    xlims!(ax_xvx, xlim[1], xlim[2])
    ylims!(ax_xvx, ylim[1], ylim[2])

    fig_xy = Figure(size=(900, 900))

    title_xy = @sprintf("Periodic Orbit in x-y plane  p:q=%d:%d", ROW.p, ROW.q)

    ax_xy = Axis(fig_xy[1, 1], title=title_xy, aspect=DataAspect())

    u0_base = [ROW.x0, ROW.y0, ROW.vx0, ROW.vy0]

    prob_xy = ODEProblem(cr3bp_dynamics!, u0_base, (0.0, ROW.T), P)

    sol_xy = solve(prob_xy, Vern9();
                   reltol=1e-12,
                   abstol=1e-12,
                   save_everystep=true)

    lines!(ax_xy, sol_xy[1, :], sol_xy[2, :], color=:red, linewidth=1.5)

    scatter!(ax_xy,
             [-MU, 1.0 - MU],
             [0.0, 0.0],
             color=:blue,
             markersize=8.0)

    return fig_xy, fig_xvx
end

# ==============================================================================
# 11. Save
# ==============================================================================
function savefig(fig, filename::AbstractString)
    FileIO.save(filename, fig)
    return filename
end

end # module

# ==============================================================================
# Run
# ==============================================================================
using .GanimedePoincareManifolds

res = GanimedePoincareManifolds.main(
    n=3000,
    eps=3e-5,
    Smax=5,
    tmax=200.0,
    vy_filter=:none,

    # ポアンカレ断面上の隣接距離がこの値以上なら中点を追加
    refine_distance=0.01,

    # 今回は s=2 を描いているので、s=2 の距離で細分化判定
    # 全交差回数で判定したい場合は refine_s=:all
    refine_s=2,

    # 距離確認 → 初期値追加 → 再伝播 の繰り返し回数の上限
    max_refine_iter=10,

    # 念のため点数の増えすぎを防ぐ上限
    max_points_per_branch=30000,
)

fig_xy, fig_xvx = GanimedePoincareManifolds.plot(
    res;
    s_s_only=2,
    s_u_only=2,
    connect=true,
    scatter_all=true,
    gamma=1.0,
    xlim=(-1.1, -0.5),
    ylim=(-0.25, 0.25),
    markersize=2.8,
    linewidth=1.2,
)

GanimedePoincareManifolds.savefig(
    fig_xy,
    joinpath(GanimedePoincareManifolds.PLOT_DIR, "orbit_xy.png"),
)

GanimedePoincareManifolds.savefig(
    fig_xvx,
    joinpath(GanimedePoincareManifolds.PLOT_DIR, "manifold_xvx_adaptive.png"),
)

# display(fig_xy)
# display(fig_xvx)