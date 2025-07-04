# the function to compute z = a * x + b * y
function axpby_kernel!(a::Float64, x::CuDeviceVector{Float64}, b::Float64, y::CuDeviceVector{Float64}, z::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a::Float64, x::CuVector{Float64}, b::Float64, y::CuVector{Float64}, z::CuVector{Float64}, n::Int)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
end

# the kernel function to update z_bar and x_bar (x_hat), Steps 6 and 7 in Algorithm 2
function combined_kernel_x_z1!(x::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds z_bar[i] = x[i] + sigma * (ATy[i] - c[i])
        @inbounds x_bar[i] = z_bar[i] < l[i] ? l[i] : (z_bar[i] > u[i] ? u[i] : z_bar[i])
        @inbounds z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        @inbounds x_hat[i] = 2 * x_bar[i] - x[i]
    end
    return
end

# the kernel function to update x_bar, Steps 7 in Algorithm 2
function combined_kernel_x_z2!(x::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds x_bar[i] = x[i] + sigma * (ATy[i] - c[i])
        @inbounds x_bar[i] = x_bar[i] < l[i] ? l[i] : (x_bar[i] > u[i] ? u[i] : x_bar[i])
        @inbounds x_hat[i] = 2 * x_bar[i] - x[i]
    end
    return
end

function update_x_z_gpu!(ws::HPRLP_workspace_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    if ws.update_z
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x_z1!(ws.x, ws.z_bar, ws.x_bar, ws.x_hat, ws.l, ws.u, ws.sigma, ws.ATy, ws.c, ws.n)
    else
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x_z2!(ws.x, ws.x_bar, ws.x_hat, ws.l, ws.u, ws.sigma, ws.ATy, ws.c, ws.n)
    end
end

function update_x_z_cpu!(ws::HPRLP_workspace_cpu)
    mul!(ws.ATy, ws.AT, ws.y)
    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
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
            @inbounds z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        end
    else
        @simd for i in eachindex(x)
            @inbounds x_bar[i] = x[i] + sigma * (ATy[i] - c[i])
            @inbounds x_bar[i] = x_bar[i] < l[i] ? l[i] : (x_bar[i] > u[i] ? u[i] : x_bar[i])
            @inbounds x_hat[i] = 2 * x_bar[i] - x[i]
        end
    end
end

# the kernel function to update y_bar and y_hat, Steps 8 and 9 in Algorithm 2
function update_y_kernel!(y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    y_hat::CuDeviceVector{Float64}, 
    b::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact::Float64,
    m::Int, m1::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m1
        @inbounds y_bar[i] = y[i] + fact * (b[i] - Ax[i])
        @inbounds y_hat[i] = 2 * y_bar[i] - y[i]
    elseif i <= m
        @inbounds y_bar[i] = max(y[i] + fact * (b[i] - Ax[i]), 0.0)
        @inbounds y_hat[i] = 2 * y_bar[i] - y[i]
    end
    return
end

function update_y_gpu!(ws::HPRLP_workspace_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.y_bar, ws.y, ws.y_hat, ws.b, ws.Ax, 1 / ws.lambda_max / ws.sigma, ws.m, ws.m1)
end

function update_y_cpu!(ws::HPRLP_workspace_cpu)
    mul!(ws.Ax, ws.A, ws.x_hat)
    fact = 1.0 / (ws.lambda_max * ws.sigma)
    m1 = ws.m1
    m = ws.m
    y = ws.y
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    b = ws.b
    Ax = ws.Ax
    @simd for i in eachindex(y)
        # Threads.@threads for i in eachindex(y)
        if i <= m1
            @inbounds y_bar[i] = y[i] + fact * (b[i] - Ax[i])
            @inbounds y_hat[i] = 2 * y_bar[i] - y[i]
        elseif i <= m
            @inbounds y_bar[i] = max(y[i] + fact * (b[i] - Ax[i]), 0.0)
            @inbounds y_hat[i] = 2 * y_bar[i] - y[i]
        end
    end
end

# the kernel function to compute the dual residuals, ||c - A^T y - z||
function compute_Rd_kernel!(ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds Rd[i] = c[i] - ATy[i] - z[i]
    end
    return
end

function compute_Rd_gpu!(ws::HPRLP_workspace_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATy, ws.z_bar, ws.c, ws.Rd, ws.n)
end

function compute_err_Rd_cpu!(ws::HPRLP_workspace_cpu)
    mul!(ws.Rd, ws.AT, ws.y_bar)
    c = ws.c
    Rd = ws.Rd
    z_bar = ws.z_bar
    @simd for i in eachindex(Rd)
        # Threads.@threads for i in eachindex(Rp)
        @inbounds Rd[i] = Rd[i] + z_bar[i] - c[i]
    end
end

# the kernel function to compute the primal residuals, ||\Pi_D(b - Ax)||
function compute_err_Rp_kernel!(Rp::CuDeviceVector{Float64},
    b::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    m1::Int, m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m1
        @inbounds Rp[i] = b[i] - Ax[i]
    elseif i <= m
        @inbounds Rp[i] = max(b[i] - Ax[i], 0.0)
    end
    return
end

function compute_err_lu_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds dx[i] = (x[i] < l[i]) ? (l[i] - x[i]) : ((x[i] > u[i]) ? (x[i] - u[i]) : 0.0)
    end
    return
end

function compute_err_Rp_gpu!(ws::HPRLP_workspace_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_err_Rp_kernel!(ws.Rp, ws.b, ws.Ax, ws.m1, ws.m)
end

function compute_err_Rp_cpu!(ws::HPRLP_workspace_cpu)
    mul!(ws.Ax, ws.A, ws.x_bar)
    m1 = ws.m1
    m = ws.m
    b = ws.b
    Ax = ws.Ax
    Rp = ws.Rp
    @simd for i in eachindex(Rp)
        # Threads.@threads for i in eachindex(Rp)
        if i <= m1
            @inbounds Rp[i] = b[i] - Ax[i]
        elseif i <= m
            @inbounds Rp[i] = max(b[i] - Ax[i], 0.0)
        end
    end
end