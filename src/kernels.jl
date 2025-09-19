# the function to compute z = a * x + b * y
function axpby_kernel!(a::Float64, x::CuDeviceVector{FloatType}, b::Float64, y::CuDeviceVector{FloatType}, z::CuDeviceVector{FloatType}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a::Float64, x::CuVector{FloatType}, b::Float64, y::CuVector{FloatType}, z::CuVector{FloatType}, n::Int)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
end

function combined_kernel_x_z1!(
    x::CuDeviceVector{FloatType},
    z_bar::CuDeviceVector{FloatType},
    x_bar::CuDeviceVector{FloatType},
    x_hat::CuDeviceVector{FloatType},
    l::CuDeviceVector{FloatType},
    u::CuDeviceVector{FloatType},
    sigma::FloatType,
    ATy::CuDeviceVector{FloatType},
    c::CuDeviceVector{FloatType},
    x0::CuDeviceVector{FloatType},
    fact1::Float64,
    fact2::Float64,
    n::Int)

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            xi = x[i]
            ATy_ci = ATy[i] - c[i]
            z_temp = xi + sigma * ATy_ci
            li = l[i]
            ui = u[i]
            # branchless clamp: clamp(z_temp, li, ui)
            xbar = min(max(z_temp, li), ui)
            zbar = (xbar - z_temp) / sigma
            xhat = 2 * xbar - xi
            xnew = muladd(fact2, xhat, fact1 * x0[i])  # fused multiply-add
            z_bar[i] = zbar
            x_bar[i] = xbar
            x_hat[i] = xhat
            x[i] = xnew
        end
    end
    return
end

# the kernel function to update x_bar, Steps 7, 9, and 10 in Algorithm 2
function combined_kernel_x_z2!(
    x::CuDeviceVector{FloatType},
    x_bar::CuDeviceVector{FloatType},
    x_hat::CuDeviceVector{FloatType},
    l::CuDeviceVector{FloatType},
    u::CuDeviceVector{FloatType},
    sigma::FloatType,
    ATy::CuDeviceVector{FloatType},
    x0::CuDeviceVector{FloatType},
    c::CuDeviceVector{FloatType},
    fact1::FloatType,
    fact2::FloatType,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            xi = x[i]
            li = l[i]
            ui = u[i]
            z_temp = xi + sigma * (ATy[i] - c[i])
            xbar = min(max(z_temp, li), ui)           # branchless clamp
            xhat = 2 * xbar - xi
            xnew = muladd(fact2, xhat, fact1 * x0[i])  # fused multiply-add
            x_bar[i] = xbar
            x_hat[i] = xhat
            x[i] = xnew
        end
    end
    return
end

function update_x_z_gpu!(ws::HPRLP_workspace_gpu, fact1::Float64, fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    if ws.update_z
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x_z1!(ws.x, ws.z_bar, ws.x_bar, ws.x_hat, ws.l, ws.u, ws.sigma, ws.ATy, ws.c, ws.last_x, fact1, fact2, ws.n)
    else
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x_z2!(ws.x, ws.x_bar, ws.x_hat, ws.l, ws.u, ws.sigma, ws.ATy, ws.c, ws.last_x, fact1, fact2, ws.n)
    end
end

function update_x_z_cpu!(ws::HPRLP_workspace_cpu, fact1::FloatType, fact2::FloatType)
    mul!(ws.ATy, ws.AT, ws.y)
    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
    x0 = ws.last_x
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    ATy = ws.ATy
    c = ws.c
    if ws.update_z
        @simd for i in eachindex(x)
            @inbounds z_bar[i] = x[i] + sigma * (ATy[i] - c[i])
            @inbounds x_bar[i] = z_bar[i] < l[i] ? l[i] : (z_bar[i] > u[i] ? u[i] : z_bar[i])
            @inbounds x_hat[i] = 2 * x_bar[i] - x[i]
            @inbounds x[i] = fact1 * x0[i] + fact2 * x_hat[i]
            @inbounds z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        end
    else
        @simd for i in eachindex(x)
            @inbounds x_bar[i] = x[i] + sigma * (ATy[i] - c[i])
            @inbounds x_bar[i] = x_bar[i] < l[i] ? l[i] : (x_bar[i] > u[i] ? u[i] : x_bar[i])
            @inbounds x_hat[i] = 2 * x_bar[i] - x[i]
            @inbounds x[i] = fact1 * x0[i] + fact2 * x_hat[i]
        end
    end
end

function update_y_kernel!(
    y_bar::CuDeviceVector{FloatType},
    y::CuDeviceVector{FloatType},
    y_hat::CuDeviceVector{FloatType},
    y_obj::CuDeviceVector{FloatType},
    AL::CuDeviceVector{FloatType},
    AU::CuDeviceVector{FloatType},
    Ax::CuDeviceVector{FloatType},
    fact1::Float64,
    fact2::Float64,
    y0::CuDeviceVector{FloatType},
    Halpern_fact1::Float64,
    Halpen_fact2::Float64,
    m::Int)

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            yi = y[i]
            ai = Ax[i]
            li = AL[i]
            ui = AU[i]
            y0i = y0[i]
            v = ai - fact1 * yi
            # branchless projection difference
            d = max(li - v, min(ui - v, 0.0))
            yb = fact2 * d
            yh = 2 * yb - yi
            ynew = muladd(Halpen_fact2, yh, Halpern_fact1 * y0i)  # fused multiply-add
            y_bar[i] = yb
            y_hat[i] = yh
            y_obj[i] = v + d
            y[i] = ynew
        end
    end
    return
end

function update_y_gpu!(ws::HPRLP_workspace_gpu, Halpern_fact1::Float64, Halpen_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.y_bar, ws.y, ws.y_hat, ws.y_obj, ws.AL, ws.AU, ws.Ax, fact1, fact2, ws.last_y, Halpern_fact1, Halpen_fact2, ws.m)
end

function update_y_cpu!(ws::HPRLP_workspace_cpu, Halpern_fact1::FloatType, Halpen_fact2::FloatType)
    mul!(ws.Ax, ws.A, ws.x_hat)
    fact1 = ws.lambda_max * ws.sigma
    fact2 = 1.0 / fact1
    y = ws.y
    y_obj = ws.y_obj
    AL = ws.AL
    AU = ws.AU
    y0 = ws.last_y
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    Ax = ws.Ax
    @simd for i in eachindex(y)
        @inbounds begin
            yi = y[i]
            # scaled residual
            v = Ax[i] - fact1 * yi
            d = max(AL[i] - v, min(AU[i] - v, 0.0))
            # for computing the dual obj function value
            y_obj[i] = v + d
            # scaled update
            yb = fact2 * d
            y_bar[i] = yb
            # branchless y_hat
            y_hat[i] = 2 * yb - yi
            # Halpern update
            y[i] = Halpern_fact1 * y0[i] + Halpen_fact2 * y_hat[i]
        end
    end
    return
end

# the kernel function to compute the dual residuals, ||c - A^T y - z||
function compute_Rd_kernel!(col_norm::CuDeviceVector{FloatType},
    ATy::CuDeviceVector{FloatType},
    z::CuDeviceVector{FloatType},
    c::CuDeviceVector{FloatType},
    Rd::CuDeviceVector{FloatType},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds Rdi = c[i] - ATy[i] - z[i]
        @inbounds Rd[i] = Rdi*col_norm[i]
    end
    return
end

function compute_Rd_gpu!(ws::HPRLP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(sc.col_norm, ws.ATy, ws.z_bar, ws.c, ws.Rd, ws.n)
end

function compute_err_Rd_cpu!(ws::HPRLP_workspace_cpu, sc::Scaling_info_cpu)
    mul!(ws.Rd, ws.AT, ws.y_bar)
    c = ws.c
    Rd = ws.Rd
    z_bar = ws.z_bar
    col_norm = sc.col_norm
    @simd for i in eachindex(Rd)
        @inbounds Rd[i] = Rd[i] + z_bar[i] - c[i]
        @inbounds Rd[i] *= col_norm[i]
    end
end



function compute_err_lu_kernel!(col_norm::CuDeviceVector{FloatType},
    dx::CuDeviceVector{FloatType},
    x::CuDeviceVector{FloatType},
    l::CuDeviceVector{FloatType},
    u::CuDeviceVector{FloatType},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds dx[i] = (x[i] < l[i]) ? (l[i] - x[i]) : ((x[i] > u[i]) ? (x[i] - u[i]) : 0.0)
        @inbounds dx[i] /= col_norm[i]
    end
    return
end


# the kernel function to compute the primal residuals, ||\Pi_D(b - Ax)||
@inline function compute_err_Rp_kernel!(row_norm::CuDeviceVector{FloatType},
    Rp::CuDeviceVector{FloatType},
    AL::CuDeviceVector{FloatType}, AU::CuDeviceVector{FloatType},
    Ax::CuDeviceVector{FloatType}, m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            # load into registers
            v = Ax[i]
            low = AL[i]
            high = AU[i]
            row_normi = row_norm[i]
            Rpi = max(min(high - v, 0), low - v)
            Rp[i] = row_normi * Rpi
        end
    end
    return
end


function compute_err_Rp_gpu!(ws::HPRLP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_err_Rp_kernel!(sc.row_norm, ws.Rp, ws.AL, ws.AU, ws.Ax, ws.m)
end

function compute_err_Rp_cpu!(ws::HPRLP_workspace_cpu, sc::Scaling_info_cpu)
    mul!(ws.Ax, ws.A, ws.x_bar)
    AL = ws.AL
    AU = ws.AU
    Ax = ws.Ax
    Rp = ws.Rp
    row_norm = sc.row_norm

    # Parallelize and eliminate bounds checks & branching
    @simd for i in eachindex(Rp)
        @inbounds begin
            v = Ax[i]
            low = AL[i]
            high = AU[i]
            #   • If v < AL: AL−v > 0 and v−AU ≤ 0 → diff = AL−v
            #   • If v > AU: v−AU > 0 and AL−v ≤ 0 → diff = v−AU
            #   • Else both are ≤ 0 → diff = 0
            Rp[i] = max(min(high - v, 0), low - v)
            Rp[i] *= row_norm[i]
        end
    end
end