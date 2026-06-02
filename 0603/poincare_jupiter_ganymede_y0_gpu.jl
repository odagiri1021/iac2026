# ============================================================
# Poincare section map in Jupiter-Ganymede CR3BP
# Section: y = 0, x < 0
# Plot: x - xdot Poincare section
# CPU + GPU
# ============================================================

using LinearAlgebra
using Printf
using CairoMakie
using DataFrames, CSV
using CUDA

CUDA.allowscalar(false)

CairoMakie.activate!(type = "png")

# ------------------------ Config ------------------------

Base.@kwdef struct Config
    system::Symbol       = :jupiter_ganymede
    C_target::Float64    = 3.0068

    # initial grid on y = 0 section
    nx::Int              = 450
    nvx::Int             = 450
    x_min::Float64       = -1.60
    x_max::Float64       = -0.02
    vx_min::Float64      = -2.50
    vx_max::Float64      =  2.50

    # Poincare map
    RoundNum::Int        = 50
    RoundMaxDur::Float64 = 10π
    RelTol::Float64      = 1e-13
    AbsTol::Float64      = 1e-13
    eta::Float64         = 1 / 200

    # +1: ydot > 0 crossing only
    # -1: ydot < 0 crossing only
    #  0: both crossings
    section_direction::Int = 1
    section_xmax::Float64 = 0.0

    # initial branch
    # :positive -> initial ydot > 0
    # :negative -> initial ydot < 0
    # :both     -> both
    branch::Symbol       = :positive

    # GPU
    # 0: CPU
    # 1: force GPU
    # 2: auto GPU if available
    IS_GPU::Int          = 2
    gpu_threads::Int     = 256
    gpu_n_corr::Int      = 4
    gpu_max_steps::Int   = 300_000

    plot_save::Bool      = true
    file_stub::String    = "jupiter_ganymede_y0_C30068"
    theme_font::String   = "Times New Roman"

    fig_size_init::Tuple{Int,Int} = (900, 750)
    fig_size_xy::Tuple{Int,Int}   = (900, 750)
    fig_size_sec::Tuple{Int,Int}  = (1000, 750)

    out_dir::String      = @__DIR__
end

# --------------------- Constants & Utils ---------------------

function Constants_CR3BP(; system::Symbol = :jupiter_ganymede)
    G = 6.6743e-20 # km^3 / kg / s^2

    if system == :jupiter_ganymede
        # Anderson paper value
        mu_target = 0.0000780369094055
        M1 = 1.89813e27
        M2 = mu_target / (1 - mu_target) * M1
        D  = 1.0704e6
        R1 = 71_492.0
        R2 = 2_634.1

    elseif system == :jupiter_europa
        mu_target = 0.0000252664488504
        M1 = 1.89813e27
        M2 = mu_target / (1 - mu_target) * M1
        D  = 671_100.0
        R1 = 71_492.0
        R2 = 1_560.8

    elseif system == :earth_moon
        M1 = 5.9724e24
        M2 = 7.346e22
        D  = 3.8498e5
        R1 = 6_371.0
        R2 = 1_737.4

    else
        error("Unknown system: $system")
    end

    return G, M1, M2, D, R1, R2
end

@inline dot3(x1, y1, z1, x2, y2, z2) = x1*x2 + y1*y2 + z1*z2

function Jacobi_const(x::AbstractVector{<:Real}, mu::Real)
    r1 = sqrt((x[1] + mu)^2 + x[2]^2 + x[3]^2)
    r2 = sqrt((x[1] - 1 + mu)^2 + x[2]^2 + x[3]^2)
    U  = 0.5 * (x[1]^2 + x[2]^2) + (1 - mu) / r1 + mu / r2
    v2 = x[4]^2 + x[5]^2 + x[6]^2
    return 2U - v2
end

# --------------------- CR3BP dynamics CPU ---------------------

function DynamicsEquation_CR3BP_Hermite(x, y, z, vx, vy, vz, mu)
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)

    Ωx = x - (1 - mu)*(x + mu)/r1^3 - mu*(x - 1 + mu)/r2^3
    Ωy = y - (1 - mu)*y/r1^3       - mu*y/r2^3
    Ωz =   - (1 - mu)*z/r1^3       - mu*z/r2^3

    ax =  2 * vy + Ωx
    ay = -2 * vx + Ωy
    az =          Ωz

    dot_r1v = (x + mu)*vx + y*vy + z*vz
    dot_r2v = (x - 1 + mu)*vx + y*vy + z*vz

    aax = -(1 - mu) * (vx/r1^3 - 3*dot_r1v*(x + mu)/r1^5) -
          mu       * (vx/r2^3 - 3*dot_r2v*(x - 1 + mu)/r2^5) + vx + 2*ay

    aay = -(1 - mu) * (vy/r1^3 - 3*dot_r1v*y/r1^5) -
          mu       * (vy/r2^3 - 3*dot_r2v*y/r2^5)             + vy - 2*ax

    aaz = -(1 - mu) * (vz/r1^3 - 3*dot_r1v*z/r1^5) -
          mu       * (vz/r2^3 - 3*dot_r2v*z/r2^5)

    return ax, ay, az, aax, aay, aaz
end

function Cal_1step_CRTBP(x0, y0, z0, vx0, vy0, vz0, dt, mu, RelTol, AbsTol)
    ax0, ay0, az0, dax0, day0, daz0 =
        DynamicsEquation_CR3BP_Hermite(x0, y0, z0, vx0, vy0, vz0, mu)

    xp  = x0 + dt*vx0 + (dt^2/2)*ax0 + (dt^3/6)*dax0
    yp  = y0 + dt*vy0 + (dt^2/2)*ay0 + (dt^3/6)*day0
    zp  = z0 + dt*vz0 + (dt^2/2)*az0 + (dt^3/6)*daz0

    vxp = vx0 + dt*ax0 + (dt^2/2)*dax0
    vyp = vy0 + dt*ay0 + (dt^2/2)*day0
    vzp = vz0 + dt*az0 + (dt^2/2)*daz0

    xt, yt, zt = xp, yp, zp
    vxt, vyt, vzt = vxp, vyp, vzp

    while true
        ax1, ay1, az1, dax1, day1, daz1 =
            DynamicsEquation_CR3BP_Hermite(xt, yt, zt, vxt, vyt, vzt, mu)

        ddax0  = (-6*(ax0 - ax1) - dt*(4*dax0 + 2*dax1)) / dt^2
        dday0  = (-6*(ay0 - ay1) - dt*(4*day0 + 2*day1)) / dt^2
        ddaz0  = (-6*(az0 - az1) - dt*(4*daz0 + 2*daz1)) / dt^2

        dddax0 = ( 12*(ax0 - ax1) + 6*dt*(dax0 + dax1)) / dt^3
        ddday0 = ( 12*(ay0 - ay1) + 6*dt*(day0 + day1)) / dt^3
        dddaz0 = ( 12*(az0 - az1) + 6*dt*(daz0 + daz1)) / dt^3

        xc  = xp  + (dt^4/24)*ddax0 + (dt^5/120)*dddax0
        yc  = yp  + (dt^4/24)*dday0 + (dt^5/120)*ddday0
        zc  = zp  + (dt^4/24)*ddaz0 + (dt^5/120)*dddaz0

        vxc = vxp + (dt^3/6)*ddax0 + (dt^4/24)*dddax0
        vyc = vyp + (dt^3/6)*dday0 + (dt^4/24)*ddday0
        vzc = vzp + (dt^3/6)*ddaz0 + (dt^4/24)*dddaz0

        if abs(xc-xt)   < max(RelTol*abs(xc), AbsTol) &&
           abs(yc-yt)   < max(RelTol*abs(yc), AbsTol) &&
           abs(zc-zt)   < max(RelTol*abs(zc), AbsTol) &&
           abs(vxc-vxt) < max(RelTol*abs(vxc), AbsTol) &&
           abs(vyc-vyt) < max(RelTol*abs(vyc), AbsTol) &&
           abs(vzc-vzt) < max(RelTol*abs(vzc), AbsTol)

            a2 = sqrt(ddax0^2 + dday0^2 + ddaz0^2)
            a3 = sqrt(dddax0^2 + ddday0^2 + dddaz0^2)
            return xc, yc, zc, vxc, vyc, vzc, a2, a3
        end

        xt, yt, zt = xc, yc, zc
        vxt, vyt, vzt = vxc, vyc, vzc
    end
end

function Hermite_CR3BP_y0_section(t0, t1,
                                  x0, y0, z0, vx0, vy0, vz0,
                                  RelTol, AbsTol, eta, mu, DU, TU,
                                  R1_km, R2_km,
                                  section_direction::Int,
                                  section_xmax::Float64)

    t  = t0
    xt, yt, zt = x0, y0, z0
    vxt, vyt, vzt = vx0, vy0, vz0

    Rlim1 = R1_km / DU
    Rlim2 = R2_km / DU
    dtlim = 0.1 * 24 * 60 * 60 / TU

    first = true
    a2_prev = 0.0
    a3_prev = 0.0

    while abs(t - t0) < abs(t1 - t0)
        r1_now = sqrt((xt + mu)^2 + yt^2 + zt^2)
        r2_now = sqrt((xt - 1 + mu)^2 + yt^2 + zt^2)

        if r1_now < Rlim1 || r2_now < Rlim2
            return false, NaN, NaN, NaN, NaN, NaN, NaN, NaN
        end

        ax, ay, az, dax, day, daz =
            DynamicsEquation_CR3BP_Hermite(xt, yt, zt, vxt, vyt, vzt, mu)

        a  = sqrt(ax^2 + ay^2 + az^2)
        da = sqrt(dax^2 + day^2 + daz^2)

        if first
            dt = da > 0 ? eta * a / da : dtlim
        else
            if da < 0.01
                den = da * a3_prev + a2_prev^2
                num = a * a2_prev + da^2
                dt = den > 0 ? eta * sqrt(num / den) : dtlim
            else
                dt = da > 0 ? eta * a / da : dtlim
            end
        end

        dt = min(dt, dtlim)

        if (t < t1) && (t + dt > t1)
            dt = t1 - t
        elseif (t > t1) && (t + dt < t1)
            dt = t1 - t
        end

        told = t
        xold, yold, zold = xt, yt, zt
        vxold, vyold, vzold = vxt, vyt, vzt

        xc, yc, zc, vxc, vyc, vzc, a2_prev, a3_prev =
            Cal_1step_CRTBP(xt, yt, zt, vxt, vyt, vzt, dt, mu, RelTol, AbsTol)

        if !first && yold * yc < 0
            cross_dir = yc - yold > 0 ? 1 : -1
            dir_ok = section_direction == 0 || cross_dir == section_direction

            α = -yold / (yc - yold)

            xhit  = xold  + α * (xc  - xold)
            yhit  = 0.0
            zhit  = zold  + α * (zc  - zold)
            vxhit = vxold + α * (vxc - vxold)
            vyhit = vyold + α * (vyc - vyold)
            vzhit = vzold + α * (vzc - vzold)
            thit  = told  + α * dt

            if dir_ok && xhit < section_xmax
                return true, thit, xhit, yhit, zhit, vxhit, vyhit, vzhit
            end
        end

        t = t + dt
        xt, yt, zt = xc, yc, zc
        vxt, vyt, vzt = vxc, vyc, vzc
        first = false

        if t >= t1
            break
        end
    end

    return false, NaN, NaN, NaN, NaN, NaN, NaN, NaN
end

# ------------------ Initial section grid ------------------

function CreateInitial_y0_section_grid(conf::Config,
                                       mu::Float64,
                                       C::Float64,
                                       R1_nd::Float64,
                                       R2_nd::Float64)

    xs  = range(conf.x_min, conf.x_max; length = conf.nx)
    vxs = range(conf.vx_min, conf.vx_max; length = conf.nvx)

    x_init = Float64[]
    y_init = Float64[]
    z_init = Float64[]
    vx_init = Float64[]
    vy_init = Float64[]
    vz_init = Float64[]

    for x in xs
        y = 0.0
        z = 0.0

        r1 = sqrt((x + mu)^2 + y^2 + z^2)
        r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)

        if r1 < R1_nd || r2 < R2_nd || x >= conf.section_xmax
            continue
        end

        U = 0.5 * (x^2 + y^2) + (1 - mu) / r1 + mu / r2

        for vx in vxs
            vz = 0.0
            vy2 = 2U - C - vx^2 - vz^2

            if vy2 < 0
                continue
            end

            vy = sqrt(vy2)

            if conf.branch === :positive
                push!(x_init, x)
                push!(y_init, y)
                push!(z_init, z)
                push!(vx_init, vx)
                push!(vy_init, vy)
                push!(vz_init, vz)

            elseif conf.branch === :negative
                push!(x_init, x)
                push!(y_init, y)
                push!(z_init, z)
                push!(vx_init, vx)
                push!(vy_init, -vy)
                push!(vz_init, vz)

            elseif conf.branch === :both
                push!(x_init, x)
                push!(y_init, y)
                push!(z_init, z)
                push!(vx_init, vx)
                push!(vy_init, vy)
                push!(vz_init, vz)

                push!(x_init, x)
                push!(y_init, y)
                push!(z_init, z)
                push!(vx_init, vx)
                push!(vy_init, -vy)
                push!(vz_init, vz)
            else
                error("branch must be :positive, :negative, or :both")
            end
        end
    end

    return x_init, y_init, z_init, vx_init, vy_init, vz_init
end

# ------------------ Forbidden region on y=0 section ------------------

function CalcForbiddenBoundary_xvx(conf::Config, mu::Float64, C::Float64; nx_plot::Int = 2000)
    xs = collect(range(conf.x_min, conf.x_max; length = nx_plot))

    vx_plus  = fill(NaN, nx_plot)
    vx_minus = fill(NaN, nx_plot)

    for i in eachindex(xs)
        x = xs[i]
        y = 0.0
        z = 0.0

        r1 = sqrt((x + mu)^2 + y^2 + z^2)
        r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)

        rhs = x^2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - C

        if rhs >= 0
            v = sqrt(rhs)
            vx_plus[i]  =  v
            vx_minus[i] = -v
        end
    end

    return xs, vx_plus, vx_minus
end

# ------------------ CPU batch ------------------

function ParallelSection_CPU(RoundNum, RoundMaxDur,
                             x0v, y0v, z0v, vx0v, vy0v, vz0v,
                             Problem, options)

    mu = Problem.mu
    DU = Problem.DU
    TU = Problem.TU

    RelTol = options.RelTol
    AbsTol = options.AbsTol
    eta    = options.eta

    section_direction = options.section_direction
    section_xmax = options.section_xmax

    _, _, _, _, R1_km, R2_km = Constants_CR3BP(; system = Problem.system)

    m_sec = Array{Float64}(undef, 0, 9)

    arr_t0  = zeros(length(x0v))
    arr_x0  = copy(x0v)
    arr_y0  = copy(y0v)
    arr_z0  = copy(z0v)
    arr_vx0 = copy(vx0v)
    arr_vy0 = copy(vy0v)
    arr_vz0 = copy(vz0v)
    arr_id  = collect(1:length(x0v))

    for round_id in 1:RoundNum
        N = length(arr_x0)

        @printf("CPU round %3d / %3d : %8d active ... ", round_id, RoundNum, N)

        if N == 0
            println("empty")
            break
        end

        arr_t1 = arr_t0 .+ RoundMaxDur

        tf  = fill(NaN, N)
        xf  = fill(NaN, N)
        yf  = fill(NaN, N)
        zf  = fill(NaN, N)
        vxf = fill(NaN, N)
        vyf = fill(NaN, N)
        vzf = fill(NaN, N)

        @inbounds for j in 1:N
            hit, tfj, xfj, yfj, zfj, vxfj, vyfj, vzfj =
                Hermite_CR3BP_y0_section(arr_t0[j], arr_t1[j],
                                          arr_x0[j], arr_y0[j], arr_z0[j],
                                          arr_vx0[j], arr_vy0[j], arr_vz0[j],
                                          RelTol, AbsTol, eta,
                                          mu, DU, TU, R1_km, R2_km,
                                          section_direction,
                                          section_xmax)

            if hit
                tf[j]  = tfj
                xf[j]  = xfj
                yf[j]  = yfj
                zf[j]  = zfj
                vxf[j] = vxfj
                vyf[j] = vyfj
                vzf[j] = vzfj
            end
        end

        keep = .!isnan.(tf)

        tf  = tf[keep]
        xf  = xf[keep]
        yf  = yf[keep]
        zf  = zf[keep]
        vxf = vxf[keep]
        vyf = vyf[keep]
        vzf = vzf[keep]
        arr_id_keep = arr_id[keep]

        if !isempty(tf)
            ids = arr_id_keep
            rounds = fill(round_id, length(tf))
            m_sec = vcat(m_sec, hcat(tf, xf, yf, zf, vxf, vyf, vzf, ids, rounds))
        end

        @printf("hits this round: %6d, accumulated: %8d\n", length(tf), size(m_sec, 1))

        if isempty(tf)
            arr_t0  = Float64[]
            arr_x0  = Float64[]
            arr_y0  = Float64[]
            arr_z0  = Float64[]
            arr_vx0 = Float64[]
            arr_vy0 = Float64[]
            arr_vz0 = Float64[]
            arr_id  = Int[]
        else
            arr_t0  = tf
            arr_x0  = xf
            arr_y0  = yf
            arr_z0  = zf
            arr_vx0 = vxf
            arr_vy0 = vyf
            arr_vz0 = vzf
            arr_id  = arr_id_keep
        end
    end

    if size(m_sec, 1) == 0
        return Float64[], Array{Float64}(undef, 0, 6), Int[], Int[]
    end

    return m_sec[:, 1],
           m_sec[:, 2:7],
           Vector{Int}(round.(Int, m_sec[:, 8])),
           Vector{Int}(round.(Int, m_sec[:, 9]))
end

# ============================================================
# GPU device functions
# ============================================================

@inline function dynamics_dev(x, y, z, vx, vy, vz, mu)
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)

    Ωx = x - (1 - mu)*(x + mu)/r1^3 - mu*(x - 1 + mu)/r2^3
    Ωy = y - (1 - mu)*y/r1^3       - mu*y/r2^3
    Ωz =   - (1 - mu)*z/r1^3       - mu*z/r2^3

    ax =  2 * vy + Ωx
    ay = -2 * vx + Ωy
    az =          Ωz

    dot_r1v = (x + mu)*vx + y*vy + z*vz
    dot_r2v = (x - 1 + mu)*vx + y*vy + z*vz

    aax = -(1 - mu) * (vx/r1^3 - 3*dot_r1v*(x + mu)/r1^5) -
          mu       * (vx/r2^3 - 3*dot_r2v*(x - 1 + mu)/r2^5) + vx + 2*ay

    aay = -(1 - mu) * (vy/r1^3 - 3*dot_r1v*y/r1^5) -
          mu       * (vy/r2^3 - 3*dot_r2v*y/r2^5)             + vy - 2*ax

    aaz = -(1 - mu) * (vz/r1^3 - 3*dot_r1v*z/r1^5) -
          mu       * (vz/r2^3 - 3*dot_r2v*z/r2^5)

    return ax, ay, az, aax, aay, aaz
end

@inline function hermite_1step_dev(x0, y0, z0, vx0, vy0, vz0,
                                   dt, mu, RelTol, AbsTol,
                                   n_corr::Int32)

    ax0, ay0, az0, dax0, day0, daz0 = dynamics_dev(x0, y0, z0, vx0, vy0, vz0, mu)

    xp  = x0 + dt*vx0 + (dt^2/2)*ax0 + (dt^3/6)*dax0
    yp  = y0 + dt*vy0 + (dt^2/2)*ay0 + (dt^3/6)*day0
    zp  = z0 + dt*vz0 + (dt^2/2)*az0 + (dt^3/6)*daz0

    vxp = vx0 + dt*ax0 + (dt^2/2)*dax0
    vyp = vy0 + dt*ay0 + (dt^2/2)*day0
    vzp = vz0 + dt*az0 + (dt^2/2)*daz0

    xt, yt, zt = xp, yp, zp
    vxt, vyt, vzt = vxp, vyp, vzp

    a2 = 0.0
    a3 = 0.0

    for it = Int32(1):n_corr
        ax1, ay1, az1, dax1, day1, daz1 = dynamics_dev(xt, yt, zt, vxt, vyt, vzt, mu)

        ddax0  = (-6*(ax0 - ax1) - dt*(4*dax0 + 2*dax1)) / dt^2
        dday0  = (-6*(ay0 - ay1) - dt*(4*day0 + 2*day1)) / dt^2
        ddaz0  = (-6*(az0 - az1) - dt*(4*daz0 + 2*daz1)) / dt^2

        dddax0 = ( 12*(ax0 - ax1) + 6*dt*(dax0 + dax1)) / dt^3
        ddday0 = ( 12*(ay0 - ay1) + 6*dt*(day0 + day1)) / dt^3
        dddaz0 = ( 12*(az0 - az1) + 6*dt*(daz0 + daz1)) / dt^3

        xc  = xp  + (dt^4/24)*ddax0 + (dt^5/120)*dddax0
        yc  = yp  + (dt^4/24)*dday0 + (dt^5/120)*ddday0
        zc  = zp  + (dt^4/24)*ddaz0 + (dt^5/120)*dddaz0

        vxc = vxp + (dt^3/6)*ddax0 + (dt^4/24)*dddax0
        vyc = vyp + (dt^3/6)*dday0 + (dt^4/24)*ddday0
        vzc = vzp + (dt^3/6)*ddaz0 + (dt^4/24)*dddaz0

        xt, yt, zt = xc, yc, zc
        vxt, vyt, vzt = vxc, vyc, vzc

        a2 = sqrt(ddax0^2 + dday0^2 + ddaz0^2)
        a3 = sqrt(dddax0^2 + ddday0^2 + dddaz0^2)
    end

    return xt, yt, zt, vxt, vyt, vzt, a2, a3
end

@inline function hermite_y0_section_dev(t0, t1,
                                        x0, y0, z0, vx0, vy0, vz0,
                                        RelTol, AbsTol, eta, mu,
                                        Rlim1, Rlim2, dtlim,
                                        section_direction::Int32,
                                        section_xmax,
                                        n_corr::Int32,
                                        max_steps::Int32)

    t  = t0
    xt, yt, zt = x0, y0, z0
    vxt, vyt, vzt = vx0, vy0, vz0

    first = true
    a2_prev = 0.0
    a3_prev = 0.0

    for step = Int32(1):max_steps
        r1_now = sqrt((xt + mu)^2 + yt^2 + zt^2)
        r2_now = sqrt((xt - 1 + mu)^2 + yt^2 + zt^2)

        if r1_now < Rlim1 || r2_now < Rlim2
            return false, NaN, NaN, NaN, NaN, NaN, NaN, NaN
        end

        ax, ay, az, dax, day, daz = dynamics_dev(xt, yt, zt, vxt, vyt, vzt, mu)

        a  = sqrt(ax^2 + ay^2 + az^2)
        da = sqrt(dax^2 + day^2 + daz^2)

        dt = dtlim

        if first
            dt = da > 0 ? eta * a / da : dtlim
        else
            if da < 0.01
                den = da * a3_prev + a2_prev^2
                num = a * a2_prev + da^2
                dt = den > 0 ? eta * sqrt(num / den) : dtlim
            else
                dt = da > 0 ? eta * a / da : dtlim
            end
        end

        dt = min(dt, dtlim)

        if t + dt > t1
            dt = t1 - t
        end

        if dt == 0
            break
        end

        told = t
        xold, yold, zold = xt, yt, zt
        vxold, vyold, vzold = vxt, vyt, vzt

        xc, yc, zc, vxc, vyc, vzc, a2_prev, a3_prev =
            hermite_1step_dev(xt, yt, zt, vxt, vyt, vzt,
                              dt, mu, RelTol, AbsTol, n_corr)

        if !first && yold * yc < 0
            cross_dir = yc - yold > 0 ? Int32(1) : Int32(-1)
            dir_ok = section_direction == Int32(0) || cross_dir == section_direction

            α = -yold / (yc - yold)

            xhit  = xold  + α * (xc  - xold)
            yhit  = 0.0
            zhit  = zold  + α * (zc  - zold)
            vxhit = vxold + α * (vxc - vxold)
            vyhit = vyold + α * (vyc - vyold)
            vzhit = vzold + α * (vzc - vzold)
            thit  = told  + α * dt

            if dir_ok && xhit < section_xmax
                return true, thit, xhit, yhit, zhit, vxhit, vyhit, vzhit
            end
        end

        t = t + dt
        xt, yt, zt = xc, yc, zc
        vxt, vyt, vzt = vxc, vyc, vzc
        first = false

        if t >= t1
            break
        end
    end

    return false, NaN, NaN, NaN, NaN, NaN, NaN, NaN
end

function section_round_kernel!(tstore, xstore, ystore, zstore,
                               vxstore, vystore, vzstore,
                               tcur, xcur, ycur, zcur,
                               vxcur, vycur, vzcur,
                               active,
                               round_id::Int32,
                               RoundMaxDur, mu,
                               Rlim1, Rlim2, dtlim,
                               RelTol, AbsTol, eta,
                               section_direction::Int32,
                               section_xmax,
                               n_corr::Int32,
                               max_steps::Int32)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(xcur)

    if i > N
        return
    end

    out = (Int(round_id) - 1) * N + i

    if !active[i]
        tstore[out] = NaN
        xstore[out] = NaN
        ystore[out] = NaN
        zstore[out] = NaN
        vxstore[out] = NaN
        vystore[out] = NaN
        vzstore[out] = NaN
        return
    end

    t0 = tcur[i]
    t1 = t0 + RoundMaxDur

    hit, tf, xf, yf, zf, vxf, vyf, vzf =
        hermite_y0_section_dev(t0, t1,
                               xcur[i], ycur[i], zcur[i],
                               vxcur[i], vycur[i], vzcur[i],
                               RelTol, AbsTol, eta, mu,
                               Rlim1, Rlim2, dtlim,
                               section_direction,
                               section_xmax,
                               n_corr,
                               max_steps)

    if hit
        tstore[out] = tf
        xstore[out] = xf
        ystore[out] = yf
        zstore[out] = zf
        vxstore[out] = vxf
        vystore[out] = vyf
        vzstore[out] = vzf

        tcur[i] = tf
        xcur[i] = xf
        ycur[i] = yf
        zcur[i] = zf
        vxcur[i] = vxf
        vycur[i] = vyf
        vzcur[i] = vzf
    else
        active[i] = false
        tstore[out] = NaN
        xstore[out] = NaN
        ystore[out] = NaN
        zstore[out] = NaN
        vxstore[out] = NaN
        vystore[out] = NaN
        vzstore[out] = NaN
    end

    return
end

function ParallelSection_GPU(RoundNum, RoundMaxDur,
                             x0v, y0v, z0v, vx0v, vy0v, vz0v,
                             Problem, options;
                             threads::Int = 256,
                             n_corr::Int = 4,
                             max_steps::Int = 300_000)

    @assert CUDA.functional() "CUDA is not functional."

    mu = Problem.mu
    DU = Problem.DU
    TU = Problem.TU

    RelTol = options.RelTol
    AbsTol = options.AbsTol
    eta    = options.eta

    _, _, _, _, R1_km, R2_km = Constants_CR3BP(; system = Problem.system)

    Rlim1 = R1_km / DU
    Rlim2 = R2_km / DU
    dtlim = 0.1 * 24 * 60 * 60 / TU

    N = length(x0v)

    if N == 0
        return Float64[], Array{Float64}(undef, 0, 6), Int[], Int[]
    end

    xcur  = CuArray(x0v)
    ycur  = CuArray(y0v)
    zcur  = CuArray(z0v)
    vxcur = CuArray(vx0v)
    vycur = CuArray(vy0v)
    vzcur = CuArray(vz0v)
    tcur  = CUDA.zeros(Float64, N)
    active = CUDA.fill(true, N)

    L = N * RoundNum

    tstore  = CUDA.fill(NaN, L)
    xstore  = CUDA.fill(NaN, L)
    ystore  = CUDA.fill(NaN, L)
    zstore  = CUDA.fill(NaN, L)
    vxstore = CUDA.fill(NaN, L)
    vystore = CUDA.fill(NaN, L)
    vzstore = CUDA.fill(NaN, L)

    blocks = cld(N, threads)

    total_hits = 0
    t_start = time()

    for round_id in 1:RoundNum
        @cuda threads=threads blocks=blocks section_round_kernel!(
            tstore, xstore, ystore, zstore,
            vxstore, vystore, vzstore,
            tcur, xcur, ycur, zcur,
            vxcur, vycur, vzcur,
            active,
            Int32(round_id),
            RoundMaxDur, mu,
            Rlim1, Rlim2, dtlim,
            RelTol, AbsTol, eta,
            Int32(options.section_direction),
            options.section_xmax,
            Int32(n_corr),
            Int32(max_steps)
        )

        CUDA.synchronize()

        lo = (round_id - 1) * N + 1
        hi = round_id * N

        t_round = Array(@view tstore[lo:hi])
        hits_round = count(.!isnan.(t_round))
        total_hits += hits_round

        n_active = count(Array(active))
        elapsed = time() - t_start

        @printf("GPU round %3d / %3d | active = %8d / %8d | hits = %8d | total = %10d | elapsed = %.1f s\n",
                round_id, RoundNum, n_active, N, hits_round, total_hits, elapsed)

        if n_active == 0
            println("All trajectories became inactive.")
            break
        end
    end

    t_h  = Array(tstore)
    x_h  = Array(xstore)
    y_h  = Array(ystore)
    z_h  = Array(zstore)
    vx_h = Array(vxstore)
    vy_h = Array(vystore)
    vz_h = Array(vzstore)

    mask = .!isnan.(t_h)

    if !any(mask)
        return Float64[], Array{Float64}(undef, 0, 6), Int[], Int[]
    end

    idx = findall(mask)

    t_sec = t_h[mask]
    X_sec = hcat(x_h[mask], y_h[mask], z_h[mask],
                 vx_h[mask], vy_h[mask], vz_h[mask])

    ids = ((idx .- 1) .% N) .+ 1
    rounds = ((idx .- 1) .÷ N) .+ 1

    return t_sec, X_sec, ids, rounds
end


# ------------------ Save one-step map pairs ------------------

@inline function Jacobi_const_vals(x, y, z, vx, vy, vz, mu)
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)
    U  = 0.5 * (x^2 + y^2) + (1 - mu) / r1 + mu / r2
    v2 = vx^2 + vy^2 + vz^2
    return 2U - v2
end

function write_map_pairs_csv(map_csv_path,
                             x0v, y0v, z0v, vx0v, vy0v, vz0v,
                             t_sec, X_sec, ids, rounds,
                             mu)

    N0 = length(x0v)

    if isempty(rounds)
        open(map_csv_path, "w") do io
            println(io, "id,n_from,n_to,t_from,x_from,y_from,z_from,vx_from,vy_from,vz_from,C_from,t_to,x_to,y_to,z_to,vx_to,vy_to,vz_to,C_to,dt,dx,dy,dz,dvx,dvy,dvz")
        end
        return 0
    end

    maxround = maximum(rounds)

    # hit_index[id, round] = row index in X_sec
    hit_index = fill(0, N0, maxround)

    @inbounds for i in eachindex(ids)
        id = ids[i]
        r  = rounds[i]
        if 1 <= id <= N0 && 1 <= r <= maxround
            hit_index[id, r] = i
        end
    end

    pair_count = 0

    open(map_csv_path, "w") do io
        println(io, "id,n_from,n_to,t_from,x_from,y_from,z_from,vx_from,vy_from,vz_from,C_from,t_to,x_to,y_to,z_to,vx_to,vy_to,vz_to,C_to,dt,dx,dy,dz,dvx,dvy,dvz")

        @inbounds for id in 1:N0
            # ------------------------------------------------------------
            # round 0 -> round 1
            # initial section point s_0 maps to first return s_1
            # ------------------------------------------------------------
            idx_to = hit_index[id, 1]

            if idx_to != 0
                t_from  = 0.0
                x_from  = x0v[id]
                y_from  = y0v[id]
                z_from  = z0v[id]
                vx_from = vx0v[id]
                vy_from = vy0v[id]
                vz_from = vz0v[id]

                t_to  = t_sec[idx_to]
                x_to  = X_sec[idx_to, 1]
                y_to  = X_sec[idx_to, 2]
                z_to  = X_sec[idx_to, 3]
                vx_to = X_sec[idx_to, 4]
                vy_to = X_sec[idx_to, 5]
                vz_to = X_sec[idx_to, 6]

                C_from = Jacobi_const_vals(x_from, y_from, z_from, vx_from, vy_from, vz_from, mu)
                C_to   = Jacobi_const_vals(x_to,   y_to,   z_to,   vx_to,   vy_to,   vz_to,   mu)

                @printf(io,
                    "%d,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    id, 0, 1,
                    t_from, x_from, y_from, z_from, vx_from, vy_from, vz_from, C_from,
                    t_to,   x_to,   y_to,   z_to,   vx_to,   vy_to,   vz_to,   C_to,
                    t_to - t_from,
                    x_to - x_from,
                    y_to - y_from,
                    z_to - z_from,
                    vx_to - vx_from,
                    vy_to - vy_from,
                    vz_to - vz_from
                )

                pair_count += 1
            end

            # ------------------------------------------------------------
            # round r -> round r+1
            # section point s_r maps to next return s_{r+1}
            # ------------------------------------------------------------
            for r in 1:(maxround - 1)
                idx_from = hit_index[id, r]
                idx_to   = hit_index[id, r + 1]

                if idx_from != 0 && idx_to != 0
                    t_from  = t_sec[idx_from]
                    x_from  = X_sec[idx_from, 1]
                    y_from  = X_sec[idx_from, 2]
                    z_from  = X_sec[idx_from, 3]
                    vx_from = X_sec[idx_from, 4]
                    vy_from = X_sec[idx_from, 5]
                    vz_from = X_sec[idx_from, 6]

                    t_to  = t_sec[idx_to]
                    x_to  = X_sec[idx_to, 1]
                    y_to  = X_sec[idx_to, 2]
                    z_to  = X_sec[idx_to, 3]
                    vx_to = X_sec[idx_to, 4]
                    vy_to = X_sec[idx_to, 5]
                    vz_to = X_sec[idx_to, 6]

                    C_from = Jacobi_const_vals(x_from, y_from, z_from, vx_from, vy_from, vz_from, mu)
                    C_to   = Jacobi_const_vals(x_to,   y_to,   z_to,   vx_to,   vy_to,   vz_to,   mu)

                    @printf(io,
                        "%d,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                        id, r, r + 1,
                        t_from, x_from, y_from, z_from, vx_from, vy_from, vz_from, C_from,
                        t_to,   x_to,   y_to,   z_to,   vx_to,   vy_to,   vz_to,   C_to,
                        t_to - t_from,
                        x_to - x_from,
                        y_to - y_from,
                        z_to - z_from,
                        vx_to - vx_from,
                        vy_to - vy_from,
                        vz_to - vz_from
                    )

                    pair_count += 1
                end
            end
        end
    end

    return pair_count
end

# ------------------------ Main ------------------------

function run(conf::Config = Config())

    mkpath(conf.out_dir)

    G, M1, M2, D, R1_km, R2_km =
        Constants_CR3BP(; system = conf.system)

    mu = M2 / (M1 + M2)
    DU = D
    TU = 1 / sqrt(G * (M1 + M2) / DU^3)

    R1_nd = R1_km / DU
    R2_nd = R2_km / DU

    C = conf.C_target

    println("------------------------------------------------------------")
    println(@sprintf("System = %s", String(conf.system)))
    println(@sprintf("Section = y = 0, x < %.3f", conf.section_xmax))
    println(@sprintf("section_direction = %d", conf.section_direction))
    println(@sprintf("Jacobi C = %.15f", C))
    println(@sprintf("mu = %.16e", mu))
    println(@sprintf("DU = %.3f km", DU))
    println(@sprintf("TU = %.6f s", TU))
    println(@sprintf("R1_nd = %.8e, R2_nd = %.8e", R1_nd, R2_nd))
    println("------------------------------------------------------------")

    x0, y0, z0, vx0, vy0, vz0 =
        CreateInitial_y0_section_grid(conf, mu, C, R1_nd, R2_nd)

    N0 = length(x0)
    println(@sprintf("Initial valid section points = %d", N0))

    set_theme!(Theme(fonts = (; regular = conf.theme_font),
                     linewidth = 1.5,
                     fontsize = 20))

    # ---------------- Initial section plot ----------------

    f0 = Figure(size = conf.fig_size_init, backgroundcolor = :white)
    ax0 = Axis(f0[1, 1], xlabel = "x [-]", ylabel = "xdot [-]")

    if N0 > 0
        scatter!(ax0, x0, vx0; markersize = 1.2, color = RGBf(0.35, 0.35, 0.35))
    end

    xlims!(ax0, conf.x_min, conf.x_max)
    ylims!(ax0, conf.vx_min, conf.vx_max)

    if conf.plot_save
        save(joinpath(conf.out_dir, "$(conf.file_stub)_initial_x_vx.png"), f0; px_per_unit = 3)
    end

    # ---------------- Propagation ----------------

    Problem = (;
        Name = "CR3BP",
        system = conf.system,
        mu,
        DU,
        TU
    )

    options = (;
        RelTol = conf.RelTol,
        AbsTol = conf.AbsTol,
        eta = conf.eta,
        section_direction = conf.section_direction,
        section_xmax = conf.section_xmax
    )

    use_gpu = false

    if conf.IS_GPU == 1
        use_gpu = true
    elseif conf.IS_GPU == 2
        use_gpu = CUDA.functional()
    else
        use_gpu = false
    end

    t_sec = Float64[]
    X_sec = Array{Float64}(undef, 0, 6)
    ids = Int[]
    rounds = Int[]

    if use_gpu
        println("Mode = GPU")
        t_sec, X_sec, ids, rounds =
            ParallelSection_GPU(conf.RoundNum,
                                conf.RoundMaxDur,
                                x0, y0, z0,
                                vx0, vy0, vz0,
                                Problem,
                                options;
                                threads = conf.gpu_threads,
                                n_corr = conf.gpu_n_corr,
                                max_steps = conf.gpu_max_steps)
    else
        println("Mode = CPU")
        t_sec, X_sec, ids, rounds =
            ParallelSection_CPU(conf.RoundNum,
                                conf.RoundMaxDur,
                                x0, y0, z0,
                                vx0, vy0, vz0,
                                Problem,
                                options)
    end

    nrow = size(X_sec, 1)

    println(@sprintf("Total section points collected = %d", nrow))

    C_check = zeros(nrow)

    for i in 1:nrow
        C_check[i] = Jacobi_const(Vector(@view X_sec[i, 1:6]), mu)
    end

    # ---------------- XY plot ----------------

    xx = range(-1.7, 1.7; length = 1001)
    yy = range(-1.7, 1.7; length = 1001)

    Xg = repeat(collect(xx), 1, length(yy))
    Yg = repeat(reshape(collect(yy), 1, :), length(xx), 1)

    r1g = sqrt.((Xg .+ mu).^2 .+ Yg.^2)
    r2g = sqrt.((Xg .- (1 - mu)).^2 .+ Yg.^2)

    U = 0.5 .* (Xg.^2 .+ Yg.^2) .+
        (1 - mu) ./ r1g .+
        mu ./ r2g

    Cmat = 2 .* U

    f1 = Figure(size = conf.fig_size_xy, backgroundcolor = :white)
    ax1 = Axis(f1[1, 1], xlabel = "x [-]", ylabel = "y [-]")

    contour!(ax1, xx, yy, -Cmat; levels = [-C], color = :black)

    if nrow > 0
        scatter!(ax1, X_sec[:, 1], X_sec[:, 2];
                 markersize = 1.2,
                 color = RGBf(0.35, 0.35, 0.35))
    end

    scatter!(ax1, [-mu], [0.0]; markersize = 12, color = :blue)
    scatter!(ax1, [1 - mu], [0.0]; markersize = 10, color = RGBf(0.93, 0.69, 0.13))

    xlims!(ax1, -1.7, 1.7)
    ylims!(ax1, -1.7, 1.7)

    if conf.plot_save
        save(joinpath(conf.out_dir, "$(conf.file_stub)_xy.png"), f1; px_per_unit = 3)
    end

    # ---------------- x - xdot Poincare section ----------------

    # ---------------- x - xdot Poincare section with forbidden region ----------------

    f2 = Figure(size = conf.fig_size_sec, backgroundcolor = :white)
    ax2 = Axis(f2[1, 1], xlabel = "x [-]", ylabel = "xdot [-]")

    xs_forbid, vx_plus, vx_minus = CalcForbiddenBoundary_xvx(conf, mu, C)

    valid = .!isnan.(vx_plus)
    xs_valid = xs_forbid[valid]
    vp = vx_plus[valid]
    vm = vx_minus[valid]

    if !isempty(xs_valid)
        # 上側の禁止領域
        band!(ax2, xs_valid, vp, fill(conf.vx_max, length(xs_valid));
            color = (:lightgray, 0.8))

        # 下側の禁止領域
        band!(ax2, xs_valid, fill(conf.vx_min, length(xs_valid)), vm;
            color = (:lightgray, 0.8))

        # 0速度曲線の境界
        lines!(ax2, xs_valid, vp; color = :black, linewidth = 2)
        lines!(ax2, xs_valid, vm; color = :black, linewidth = 2)
    end

    if nrow > 0
        scatter!(ax2, X_sec[:, 1], X_sec[:, 4];
                markersize = 1.2,
                color = RGBf(0.25, 0.25, 0.25))
    end

    xlims!(ax2, conf.x_min, conf.x_max)
    ylims!(ax2, conf.vx_min, conf.vx_max)

    if conf.plot_save
        save(joinpath(conf.out_dir, "$(conf.file_stub)_section_x_vx.png"), f2; px_per_unit = 3)
    end

    # Headless server: figures are saved to PNG, not displayed.

    # ---------------- CSV output ----------------

    if nrow > 0
        df = DataFrame(
            id = ids,
            round = rounds,
            t = t_sec,
            x = X_sec[:, 1],
            y = X_sec[:, 2],
            z = X_sec[:, 3],
            vx = X_sec[:, 4],
            vy = X_sec[:, 5],
            vz = X_sec[:, 6],
            C = C_check
        )

        csv_path = joinpath(conf.out_dir, "$(conf.file_stub)_section_points.csv")
        CSV.write(csv_path, df)

        map_csv_path = joinpath(conf.out_dir, "$(conf.file_stub)_map_pairs.csv")
        n_pairs = write_map_pairs_csv(map_csv_path,
                                      x0, y0, z0, vx0, vy0, vz0,
                                      t_sec, X_sec, ids, rounds,
                                      mu)

        println("Saved:")
        println(csv_path)
        println(map_csv_path)
        println(joinpath(conf.out_dir, "$(conf.file_stub)_initial_x_vx.png"))
        println(joinpath(conf.out_dir, "$(conf.file_stub)_xy.png"))
        println(joinpath(conf.out_dir, "$(conf.file_stub)_section_x_vx.png"))
        println(@sprintf("Map pairs saved = %d", n_pairs))
    end

    println(@sprintf("Done. Jacobi C = %.15f", C))

    return nothing
end

# ------------------------ Execute ------------------------

run(Config(;
    system = :jupiter_ganymede,
    C_target = 3.0068,

    nx = 450,
    nvx = 450,

    x_min = -1.60,
    x_max = -0.02,
    vx_min = -2.50,
    vx_max = 2.50,

    RoundNum = 50,
    RoundMaxDur = 10π,

    # Anderson型の one-sided Poincare map に近い設定
    # y = 0, x < 0, ydot > 0
    section_direction = 1,
    section_xmax = 0.0,
    branch = :positive,

    IS_GPU = 2,
    gpu_threads = 256,
    gpu_n_corr = 4,
    gpu_max_steps = 300_000,

    plot_save = true,
    file_stub = "jupiter_ganymede_y0_C30068"
))