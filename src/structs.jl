#=

Data Structures
	•	HPRLP_results → Stores solver outputs (iterations, runtime, primal/dual objectives, residuals, status).
	•	HPRLP_workspace_gpu → Holds GPU-accelerated variables (x, y, z), matrices (A, AT), and intermediate computations.
	•	HPRLP_residuals → Tracks primal/dual feasibility and duality gap for convergence checks.
	•	HPRLP_restart → Manages adaptive/fixed restart strategies.
	•	LP_info_cpu / LP_info_gpu → Stores LP problem data in CPU/GPU memory.
	•	Scaling_info → Applies problem scaling for numerical stability.
	•	HPRLP_parameters → Defines solver settings (tolerance, max iterations, restart strategy).

=#

# the space for the parameters of the HPR-LP algorithm
mutable struct HPRLP_parameters
    # the stopping tolerance, default is 1e-6
    stoptol::FloatType

    # the maximum number of iterations, default is 1000
    max_iter::Int

    # the time limit in seconds, default is 3600.0
    time_limit::FloatType

    # the check interval for the residuals, default is 150
    check_iter::Int

    # whether to use the Ruiz scaling, default is true
    use_Ruiz_scaling::Bool

    # whether to use the Pock-Chambolle scaling, default is true
    use_Pock_Chambolle_scaling::Bool

    # whether to use the scaling for b and c, default is true
    use_bc_scaling::Bool

    # use GPU or not, default is true
    use_gpu::Bool

    # GPU device number, default is 0
    device_number::Int

    # whether do warm up, default is false
    warm_up::Bool

    # print frequency, print the log every print_frequency iterations, default is -1 (auto)
    print_frequency::Int

    # Default constructor
    HPRLP_parameters() = new(1e-6, typemax(Int32), 3600.0, 150, true, true, true, true, 0, false, -1)
end


# Define the results will be returned
mutable struct HPRLP_results
    # Number of iterations
    iter::Int

    # Number of iterations for the 1e-4 accuracy
    iter_4::Int

    # Number of iterations for the 1e-6 accuracy
    iter_6::Int

    # Time in seconds
    time::FloatType

    # Time in seconds for the 1e-4 accuracy
    time_4::FloatType

    # Time in seconds for the 1e-6 accuracy
    time_6::FloatType

    read_time::FloatType
    
    # Time used by power method
    power_time::FloatType

    # Primal objective value
    primal_obj::FloatType

    # Relative residuals of the primal feasibility, dual feasibility, and objective gap
    residuals::FloatType

    # Objective gap
    gap::FloatType


    # OPTIMAL, MAX_ITER or TIME_LIMIT
    # OPTIMAL: the algorithm finds the optimal solution
    # MAX_ITER: the algorithm reaches the maximum number of iterations
    # TIME_LIMIT: the algorithm reaches the time limit
    output_type::String

    # The vector x
    x::Vector{FloatType}

    # The vector y
    y::Vector{FloatType}

    # The vector z
    z::Vector{FloatType}

    # Default constructor
    HPRLP_results() = new()
end

# Define the workspace for the HPR-LP algorithm
mutable struct HPRLP_workspace_gpu
    # The vector x
    x::CuVector{FloatType}

    # The vector x_hat, corresponding to ̂x in the paper
    x_hat::CuVector{FloatType}

    # The vector x_bar, corresponding to x̄ in the paper
    x_bar::CuVector{FloatType}

    # The vector dx, mainly used to store the difference between x1 and x2
    dx::CuVector{FloatType}

    # The vector y
    y::CuVector{FloatType}

    # The vector y_hat, corresponding to ̂y in the paper
    y_hat::CuVector{FloatType}

    # The vector y_bar, corresponding to ȳ in the paper
    y_bar::CuVector{FloatType}

    # The vector y_obj, used for computing the dual objective function variable
    y_obj::CuVector{FloatType}

    # The vector dy, mainly used to store the difference between y1 and y2
    dy::CuVector{FloatType}

    # The vector z_bar, corresponding to z̄ in the paper
    z_bar::CuVector{FloatType}

    # The sparse matrix A, corresponding to A in the paper, the constraints matrix
    A::CuSparseMatrixCSR{FloatType,Int32}

    # The sparse matrix A^T, the transpose of A
    AT::CuSparseMatrixCSR{FloatType,Int32}

    # The vector AL, the coefficients of the lower bound of the constraints
    AL::CuVector{FloatType}

    # The vector AU, the coefficients of the lower bound of the constraints
    AU::CuVector{FloatType}

    # The vector c, the coefficients of the objective function
    c::CuVector{FloatType}

    # The vector l, the lower bound of the variables
    l::CuVector{FloatType}

    # The vector u, the upper bound of the variables
    u::CuVector{FloatType}

    # The vector Rp, normally used to store the vector b-Ax
    Rp::CuVector{FloatType}

    # The vector Rd, normally used to store the vector c-A^Ty-z
    Rd::CuVector{FloatType}

    # The total number of constraints
    m::Int

    # The number of variables
    n::Int

    # The value of σ
    sigma::FloatType

    # The value of λ_max(AA^T), the maximum eigenvalue of the matrix AA^T
    lambda_max::Float64

    # Normally used to store the vector Ax
    Ax::CuVector{FloatType}

    # Normally used to store the vector ATy
    ATy::CuVector{FloatType}

    # Normally used to store the vector x that the algorithm restarted last time
    last_x::CuVector{FloatType}

    # Normally used to store the vector y that the algorithm restarted last time
    last_y::CuVector{FloatType}

    # Normally used to indicate whether the vector z should be computed
    update_z::Bool

    # Default constructor
    HPRLP_workspace_gpu() = new()
end

mutable struct HPRLP_workspace_cpu
    x::Vector{FloatType}
    x_hat::Vector{FloatType}
    x_bar::Vector{FloatType}
    dx::Vector{FloatType}
    y::Vector{FloatType}
    y_hat::Vector{FloatType}
    y_bar::Vector{FloatType}
    y_obj::Vector{FloatType}
    dy::Vector{FloatType}
    z_bar::Vector{FloatType}
    A::SparseMatrixCSC{FloatType,Int32}
    AT::SparseMatrixCSC{FloatType,Int32}
    c::Vector{FloatType}
    AL::Vector{FloatType}
    AU::Vector{FloatType}
    l::Vector{FloatType}
    u::Vector{FloatType}
    Rp::Vector{FloatType}
    Rd::Vector{FloatType}
    m::Int
    n::Int
    sigma::FloatType
    lambda_max::FloatType
    Ax::Vector{FloatType}
    ATy::Vector{FloatType}
    last_x::Vector{FloatType}
    last_y::Vector{FloatType}
    update_z::Bool
    HPRLP_workspace_cpu() = new()
end

# Define the variables related to the residuals of the HPR-LP
mutable struct HPRLP_residuals
    # The relative residuals of the primal feasibility evaluated at x_bar
    err_Rp_org_bar::FloatType

    # The relative residuals of the dual feasibility evaluated at y_bar and z_bar
    err_Rd_org_bar::FloatType

    # The primal objective value evaluated at x_bar
    primal_obj_bar::FloatType

    # The dual objective value evaluated at y_bar and z_bar
    dual_obj_bar::FloatType

    # The relative gap evaluated at x_bar, y_bar, and z_bar
    rel_gap_bar::FloatType

    # indicate whether the residuals are updated
    is_updated::Bool

    # The maximum of the primal feasibility, dual feasibility, and duality gap
    KKTx_and_gap_org_bar::FloatType

    # Define a default constructor
    HPRLP_residuals() = new()
end

# Define the variables related to the restart of the HPR-LP
mutable struct HPRLP_restart
    # indicate which restart condition is satisfied, 1: sufficient, 2: necessary, 3: long
    restart_flag::Int

    # indicate whether it is the first restart
    first_restart::Bool

    # the value \tilde{R}_{r,0}
    last_gap::FloatType

    # the value \tilde{R}_{r,t}
    current_gap::FloatType

    # the value \tilde{R}_{r,t-1}
    save_gap::FloatType

    # the best value \tilde{R}_{best}
    best_gap::FloatType

    # the  value of sigma at the best_gap
    best_sigma::FloatType

    # the number of inner iterations, t in the paper
    inner::Int

    # the number of restart step length for fixed step restart
    step::Int

    # the number of restart triggered by sufficient decrease
    sufficient::Int

    # the number of restart triggered by necessary decrease
    necessary::Int

    # the number of restart triggered by long iterations
    long::Int

    # the ratio of ||Δx|| and ||Δy||
    ratio::Int

    # the number of restart
    times::Int

    # the value of M-norm 
    weighted_norm::FloatType

    # Default constructor
    HPRLP_restart() = new()
end

# the space for the LP information on the CPU
mutable struct LP_info_cpu
    A::SparseMatrixCSC{FloatType,Int32}
    AT::SparseMatrixCSC{FloatType,Int32}
    c::Vector{FloatType}
    AL::Vector{FloatType}
    AU::Vector{FloatType}
    l::Vector{FloatType}
    u::Vector{FloatType}
    obj_constant::FloatType
end

# the space for the LP information on the GPU
mutable struct LP_info_gpu
    A::CuSparseMatrixCSR{FloatType,Int32}
    AT::CuSparseMatrixCSR{FloatType,Int32}
    c::CuVector{FloatType}
    AL::CuVector{FloatType}
    AU::CuVector{FloatType}
    l::CuVector{FloatType}
    u::CuVector{FloatType}
    obj_constant::FloatType
end

# the space for the scaling information on the CPU
mutable struct Scaling_info_cpu
    # the original vector l
    l_org::Vector{FloatType}

    # the original vector u
    u_org::Vector{FloatType}

    # the row norm of the matrix A
    row_norm::Vector{FloatType}

    # the column norm of the matrix A
    col_norm::Vector{FloatType}

    # the scaling factor for the vector b
    b_scale::FloatType

    # the scaling factor for the vector c
    c_scale::FloatType

    # the norm of the vector b
    norm_b::FloatType

    # the norm of the vector c
    norm_c::FloatType

    # the norm of the original vector b
    norm_b_org::FloatType

    # the norm of the original vector c
    norm_c_org::FloatType
end

# the space for the scaling information on the GPU
mutable struct Scaling_info_gpu
    l_org::CuVector{FloatType}
    u_org::CuVector{FloatType}
    row_norm::CuVector{FloatType}
    col_norm::CuVector{FloatType}
    b_scale::FloatType
    c_scale::FloatType
    norm_b::FloatType
    norm_c::FloatType
    norm_b_org::FloatType
    norm_c_org::FloatType
end