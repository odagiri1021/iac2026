# ============================================================
# Background Poincare map / invariant curves
# Based on the manifold code style:
#   - planar CR3BP: u = [x, y, vx, vy]
#   - Vern9()
#   - one integration per seed
#   - ContinuousCallback collects y = 0 crossings during that integration
#   - section: y = 0, x < 0
#   - plot: x - xdot
# ============================================================

module GanimedePoincareBackground

using LinearAlgebra
using Printf
using CairoMakie
using DataFrames
using CSV
using SciMLBase
using OrdinaryDiffEqVerner

CairoMakie.activate!(type = "png")

# ==============================================================================
# 0. Settings: same constants/style as the manifold code
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
const PLOT_DIR = joinpath(BASE_DIR, "plots_ganimede_background")
mkpath(PLOT_DIR)

# ==============================================================================
# 1. Parameters
# ==============================================================================
struct CR3BPParams
    μ::Float64
end

const P = CR3BPParams(MU)

Base.@kwdef struct Config
    # Use Jacobi constant of ROW by default.
    C::Float64 = jacobi_constant([ROW.x0, ROW.y0, ROW.vx0, ROW.vy0], MU)

    # Initial seed grid on Sigma: y = 0, x < 0.
    # u0 = [x, 0, vx, vy], where vy is computed from Jacobi constant.
    nx::Int = 200
    nvx::Int = 

    x_min_seed::Float64 = -1.10
    x_max_seed::Float64 = -0.50
    vx_min_seed::Float64 = -0.25
    vx_max_seed::Float64 =  0.25

    # :pos, :neg, or :both for the initial vy branch.
    initial_vy_branch::Symbol = :pos

    # Poincare section filters.
    xneg::Bool = true

    # :none plots both ydot signs.
    # :pos  plots only ydot > 0 crossings.
    # :neg  plots only ydot < 0 crossings.
    vy_filter::Symbol = :none

    # Number of crossings to collect per seed.
    Smax::Int = 80

    # Integrate each seed over this time range once.
    # This follows the reference code style and avoids restarting at each crossing.
    tmax::Float64 = 500.0

    reltol::Float64 = 1e-12
    abstol::Float64 = 1e-12
    abort_limit::Float64 = 10.0

    # Stop or discard near singular/collision regions.
    # Jupiter radius in this nondimensional system is about 0.0668.
    # Ganymede radius is about 0.00246.
    r1_min::Float64 = 0.070
    r2_min::Float64 = 0.005

    # Discard section hits if Jacobi error is larger than this.
    C_error_keep::Float64 = 1e-7

    # Plot range.
    xlim::Tuple{Float64,Float64} = (-1.20, -0.50)
    ylim::Tuple{Float64,Float64} = (-0.30,  0.30)

    markersize::Float64 = 2.5

    save_csv::Bool = true
    save_png::Bool = true

    file_stub::String = "background_poincare_Crow_Vern9"
end

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
# 3. Jacobi constant and section initial condition
# ==============================================================================
function jacobi_constant(u::AbstractVector{<:Real}, μ::Real)
    x  = u[1]
    y  = u[2]
    vx = u[3]
    vy = u[4]

    r1 = sqrt((x + μ)^2 + y^2)
    r2 = sqrt((x - 1 + μ)^2 + y^2)

    Ω = 0.5 * (x^2 + y^2) + (1.0 - μ) / r1 + μ / r2
    return 2.0 * Ω - vx^2 - vy^2
end

function vy_from_C_on_section(x::Float64,
                              vx::Float64,
                              C::Float64,
                              μ::Float64,
                              branch::Symbol)
    y = 0.0

    r1 = abs(x + μ)
    r2 = abs(x - 1.0 + μ)

    Ω = 0.5 * x^2 + (1.0 - μ) / r1 + μ / r2
    vy2 = 2.0 * Ω - C - vx^2

    if vy2 < 0.0
        return false, NaN
    end

    vy_abs = sqrt(max(vy2, 0.0))

    if branch === :pos
        return true, vy_abs
    elseif branch === :neg
        return true, -vy_abs
    else
        error("branch must be :pos or :neg")
    end
end

function make_seed_grid(conf::Config)
    seeds = Vector{Vector{Float64}}()
    seed_x = Float64[]
    seed_vx = Float64[]

    xs  = range(conf.x_min_seed, conf.x_max_seed; length = conf.nx)
    vxs = range(conf.vx_min_seed, conf.vx_max_seed; length = conf.nvx)

    branches =
        conf.initial_vy_branch === :both ? (:pos, :neg) :
        conf.initial_vy_branch === :pos  ? (:pos,) :
        conf.initial_vy_branch === :neg  ? (:neg,) :
        error("initial_vy_branch must be :pos, :neg, or :both")

    for x in xs
        if conf.xneg && !(x < 0.0)
            continue
        end

        for vx in vxs
            for br in branches
                ok, vy = vy_from_C_on_section(Float64(x), Float64(vx), conf.C, MU, br)

                if ok
                    push!(seeds, [Float64(x), 0.0, Float64(vx), vy])
                    push!(seed_x, Float64(x))
                    push!(seed_vx, Float64(vx))
                end
            end
        end
    end

    return seeds, seed_x, seed_vx
end

# ==============================================================================
# 4. Collect crossings on Sigma: y=0
#    This is intentionally close to the reference manifold code.
# ==============================================================================
function collect_poincare_hits(u0_state::Vector{Float64},
                               p::CR3BPParams;
                               tspan::Tuple{Float64,Float64},
                               Smax::Int=80,
                               xneg::Bool=true,
                               vy_filter::Symbol=:none,
                               reltol::Float64=1e-12,
                               abstol::Float64=1e-12,
                               abort_limit::Float64=10.0,
                               r1_min::Float64=0.070,
                               r2_min::Float64=0.005,
                               C_ref::Float64=NaN,
                               C_error_keep::Float64=1e-7,
                               t_ignore::Float64=1e-10,
                               dt_min::Float64=1e-8)

    xs = Float64[]
    ys = Float64[]
    vxs = Float64[]
    ts = Float64[]
    vys = Float64[]
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

        x  = integrator.u[1]
        y  = integrator.u[2]
        vx = integrator.u[3]
        vy = integrator.u[4]

        μ = p.μ
        r1 = sqrt((x + μ)^2 + y^2)
        r2 = sqrt((x - 1.0 + μ)^2 + y^2)

        if r1 < r1_min || r2 < r2_min
            terminate!(integrator)
            return
        end

        C_now = jacobi_constant([x, y, vx, vy], μ)
        if isfinite(C_ref) && abs(C_now - C_ref) > C_error_keep
            return
        end

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
        elseif vy_filter === :none
            # keep all crossings
        else
            error("vy_filter must be :none, :pos, or :neg")
        end

        last_t[] = t

        push!(ts, t)
        push!(xs, x)
        push!(ys, y)
        push!(vxs, vx)
        push!(vys, vy)

        if length(xs) >= Smax
            terminate!(integrator)
        end

        return
    end

    cb_p = ContinuousCallback(condition, affect!; save_positions=(false, false))

    cb_abort = DiscreteCallback(
        (u, t, integrator) -> begin
            μ = p.μ
            x = u[1]
            y = u[2]
            r1 = sqrt((x + μ)^2 + y^2)
            r2 = sqrt((x - 1.0 + μ)^2 + y^2)
            (abs(x) > abort_limit) || (abs(y) > abort_limit) || (r1 < r1_min) || (r2 < r2_min)
        end,
        integrator -> terminate!(integrator);
        save_positions=(false, false)
    )

    prob = ODEProblem(cr3bp_dynamics!, u0_state, tspan, p)

    solve(prob, Vern9();
          callback = CallbackSet(cb_p, cb_abort),
          reltol = reltol,
          abstol = abstol,
          save_everystep = false)

    return ts, xs, ys, vxs, vys
end

# ==============================================================================
# 5. Compute background
# ==============================================================================
function compute_background(conf::Config = Config())
    seeds, seed_x, seed_vx = make_seed_grid(conf)

    @printf("C = %.16e\n", conf.C)
    @printf("valid seeds = %d\n", length(seeds))
    @printf("grid nx x nvx = %d x %d\n", conf.nx, conf.nvx)
    @printf("Smax per seed = %d, tmax = %.3f\n", conf.Smax, conf.tmax)
    @printf("vy_filter = %s\n", String(conf.vy_filter))

    seed_id_all = Int[]
    hit_id_all = Int[]
    t_all = Float64[]
    x_all = Float64[]
    y_all = Float64[]
    vx_all = Float64[]
    vy_all = Float64[]
    C_all = Float64[]

    for sid in eachindex(seeds)
        if sid == 1 || sid % 50 == 0 || sid == length(seeds)
            @printf("seed %5d / %5d ...\n", sid, length(seeds))
            flush(stdout)
        end

        ts, xs, ys, vxs, vys = collect_poincare_hits(seeds[sid], P;
                                                     tspan = (0.0, conf.tmax),
                                                     Smax = conf.Smax,
                                                     xneg = conf.xneg,
                                                     vy_filter = conf.vy_filter,
                                                     reltol = conf.reltol,
                                                     abstol = conf.abstol,
                                                     abort_limit = conf.abort_limit,
                                                     r1_min = conf.r1_min,
                                                     r2_min = conf.r2_min,
                                                     C_ref = conf.C,
                                                     C_error_keep = conf.C_error_keep)

        for k in eachindex(xs)
            uhit = [xs[k], ys[k], vxs[k], vys[k]]

            push!(seed_id_all, sid)
            push!(hit_id_all, k)
            push!(t_all, ts[k])
            push!(x_all, xs[k])
            push!(y_all, ys[k])
            push!(vx_all, vxs[k])
            push!(vy_all, vys[k])
            push!(C_all, jacobi_constant(uhit, MU))
        end
    end

    df = DataFrame(
        seed_id = seed_id_all,
        hit = hit_id_all,
        t = t_all,
        x = x_all,
        y = y_all,
        vx = vx_all,
        vy = vy_all,
        C = C_all,
        C_error = C_all .- conf.C
    )

    seed_df = DataFrame(seed_id = collect(1:length(seeds)),
                        x0 = seed_x,
                        vx0 = seed_vx,
                        vy0 = [s[4] for s in seeds])

    return df, seed_df
end

# ==============================================================================
# 6. Plot
# ==============================================================================
function plot_background(df::DataFrame,
                         seed_df::DataFrame,
                         conf::Config = Config();
                         plot_seeds::Bool = false)

    fig = Figure(size=(1200, 900))
    ax = Axis(fig[1, 1],
              xlabel = "x [-]",
              ylabel = "xdot [-]",
              title = @sprintf("Background Poincare section, C=%.10f, crossings=%d",
                               conf.C, nrow(df)))

    if nrow(df) > 0
        scatter!(ax, df.x, df.vx;
                 markersize = conf.markersize,
                 color = RGBf(0.0, 0.0, 0.0))
    end

    if plot_seeds && nrow(seed_df) > 0
        scatter!(ax, seed_df.x0, seed_df.vx0;
                 markersize = 0.8,
                 color = RGBf(0.85, 0.0, 0.0))
    end

    xlims!(ax, conf.xlim[1], conf.xlim[2])
    ylims!(ax, conf.ylim[1], conf.ylim[2])

    return fig
end

function main(; kwargs...)
    conf = Config(; kwargs...)

    df, seed_df = compute_background(conf)

    csv_path = joinpath(PLOT_DIR, "$(conf.file_stub)_section_points.csv")
    seed_path = joinpath(PLOT_DIR, "$(conf.file_stub)_seed_points.csv")
    png_path = joinpath(PLOT_DIR, "$(conf.file_stub)_section_xvx.png")

    if conf.save_csv
        CSV.write(csv_path, df)
        CSV.write(seed_path, seed_df)
    end

    fig = plot_background(df, seed_df, conf; plot_seeds=false)

    if conf.save_png
        save(png_path, fig; px_per_unit=2)
    end

    println("Saved:")
    println(csv_path)
    println(seed_path)
    println(png_path)

    if nrow(df) > 0
        @printf("C error min = %.6e\n", minimum(df.C_error))
        @printf("C error max = %.6e\n", maximum(df.C_error))
        @printf("x  range = [%.6f, %.6f]\n", minimum(df.x), maximum(df.x))
        @printf("vx range = [%.6f, %.6f]\n", minimum(df.vx), maximum(df.vx))
    else
        println("No crossings collected.")
    end

    return df, seed_df, fig
end

end # module

using .GanimedePoincareBackground

# ==============================================================================
# Run
# ==============================================================================
GanimedePoincareBackground.main(
    nx = 20,
    nvx = 20,

    x_min_seed = -0.95,
    x_max_seed = -0.70,
    vx_min_seed = -0.18,
    vx_max_seed =  0.18,

    initial_vy_branch = :pos,

    # :none means both ydot signs are plotted.
    vy_filter = :none,

    Smax = 1200,
    tmax = 8000.0,

    reltol = 1e-12,
    abstol = 1e-12,

    xlim = (-1.20, -0.50),
    ylim = (-0.35,  0.35),

    markersize = 0.8,
)
