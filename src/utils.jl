# The function to read the LP problem from the file and formulate the LP problem
function formulation(lp)
    A = sparse(lp.arows, lp.acols, lp.avals, lp.ncon, lp.nvar)

    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall((sum(abs_A, dims=2)[:, 1] .== 0) .| ((lp.lcon .== -Inf) .& (lp.ucon .== Inf)))

    if length(del_row) > 0
        keep_rows = setdiff(1:size(A, 1), del_row)
        A = A[keep_rows, :]
        lp.lcon = lp.lcon[keep_rows]
        lp.ucon = lp.ucon[keep_rows]
        println("Deleted ", length(del_row), " rows of A that are all zeros.")
    end

    # Get the index of the different types of constraints
    idxE = findall(lp.lcon .== lp.ucon)
    idxG = findall((lp.lcon .> -Inf) .& (lp.ucon .== Inf))
    idxL = findall((lp.lcon .== -Inf) .& (lp.ucon .< Inf))
    idxB = findall((lp.lcon .> -Inf) .& (lp.ucon .< Inf))
    idxB = setdiff(idxB, idxE)

    println("problem information: nRow = ", size(A, 1), ", nCol = ", size(A, 2), ", nnz A = ", nnz(A))
    println("                     number of equalities = ", length(idxE))
    println("                     number of inequalities = ", length(idxG) + length(idxL) + length(idxB))

    @assert length(lp.lcon) == length(idxE) + length(idxG) + length(idxL) + length(idxB)

    standard_lp = LP_info_cpu(A, transpose(A), lp.c, lp.lcon, lp.ucon, lp.lvar, lp.uvar, lp.c0)

    # Return the modified lp
    return standard_lp
end

# the scaling function for the LP problem
function scaling!(lp::LP_info_cpu, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
    m, n = size(lp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    # Preallocate temporary arrays
    temp_norm1 = zeros(m)
    temp_norm2 = zeros(n)
    DA = spdiagm(temp_norm1)
    EA = spdiagm(temp_norm2)
    AL_nInf = copy(lp.AL)
    AU_nInf = copy(lp.AU)
    AL_nInf[lp.AL.==-Inf] .= 0.0
    AU_nInf[lp.AU.==Inf] .= 0.0
    norm_b_org = 1 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    norm_c_org = 1 + norm(lp.c)
    # norm_b_org = 5.42181966531778
    scaling_info = Scaling_info_cpu(copy(lp.l), copy(lp.u), row_norm, col_norm, 1, 1, 1, 1, norm_b_org, norm_c_org)
    println("norm_b_org: ", norm_b_org)
    println("norm_c_org: ", norm_c_org)
    # Ruiz scaling
    if use_Ruiz_scaling
        for _ in 1:10
            temp_norm1 .= sqrt.(maximum(abs, lp.A, dims=2)[:, 1])
            temp_norm1[iszero.(temp_norm1)] .= 1.0
            row_norm .*= temp_norm1
            DA .= spdiagm(1.0 ./ temp_norm1)
            temp_norm2 .= sqrt.(maximum(abs, lp.A, dims=1)[1, :])
            temp_norm2[iszero.(temp_norm2)] .= 1.0
            col_norm .*= temp_norm2
            EA .= spdiagm(1.0 ./ temp_norm2)
            lp.AL ./= temp_norm1
            lp.AU ./= temp_norm1
            lp.A .= DA * lp.A * EA
            lp.c ./= temp_norm2
            lp.l .*= temp_norm2
            lp.u .*= temp_norm2
        end
    end

    # Pock-Chambolle scaling
    if use_Pock_Chambolle_scaling
        temp_norm1 .= sqrt.(sum(abs, lp.A, dims=2)[:, 1])
        temp_norm1[iszero.(temp_norm1)] .= 1.0
        row_norm .*= temp_norm1
        DA .= spdiagm(1.0 ./ temp_norm1)
        temp_norm2 .= sqrt.(sum(abs, lp.A, dims=1)[1, :])
        temp_norm2[iszero.(temp_norm2)] .= 1.0
        col_norm .*= temp_norm2
        EA .= spdiagm(1.0 ./ temp_norm2)
        lp.AL ./= temp_norm1
        lp.AU ./= temp_norm1
        lp.A .= DA * lp.A * EA
        lp.c ./= temp_norm2
        lp.l .*= temp_norm2
        lp.u .*= temp_norm2
    end

    # scaling for b and c
    if use_bc_scaling
        AL_nInf = copy(lp.AL)
        AU_nInf = copy(lp.AU)
        AL_nInf[lp.AL.==-Inf] .= 0.0
        AU_nInf[lp.AU.==Inf] .= 0.0
        b_scale = 1 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
        c_scale = 1 + norm(lp.c)
        lp.AL ./= b_scale
        lp.AU ./= b_scale
        lp.c ./= c_scale
        lp.l ./= b_scale
        lp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end
    AL_nInf = copy(lp.AL)
    AU_nInf = copy(lp.AU)
    AL_nInf[lp.AL.==-Inf] .= 0.0
    AU_nInf[lp.AU.==Inf] .= 0.0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(lp.c)
    lp.AT = transpose(lp.A)
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    return scaling_info
end

# the function to compute the maximum eigenvalue of the matrix AA^T
function power_iteration_gpu(A::CuSparseMatrixCSR, AT::CuSparseMatrixCSR,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    lambda_max = 1
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q
        if CUDA.norm(q) < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", CUDA.norm(q))
    return lambda_max
end

function power_iteration_cpu(A::SparseMatrixCSC, AT::SparseMatrixCSC,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = zeros(Float64, m)
    ATq = zeros(Float64, n)
    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        mul!(ATq, AT, q)
        mul!(z, A, ATq)
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q
        if norm(q) < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", norm(q))
    return lambda_max
end

# the function to run the HPR-LP algorithm on a single file
function run_file(FILE_NAME::String, params::HPRLP_parameters)
    t_start = time()
    println("READING FILE ... ", FILE_NAME)
    io = open(FILE_NAME)
    lp = Logging.with_logger(Logging.NullLogger()) do
        readqps(io, mpsformat=:free)
    end
    close(io)
    read_time = time() - t_start
    println(@sprintf("READING FILE time: %.2f seconds", read_time))

    t_start = time()
    setup_start = time()
    println("FORMULATING LP ...")
    standard_lp = formulation(lp)
    println(@sprintf("FORMULATING LP time: %.2f seconds", time() - t_start))

    if params.use_gpu
        CUDA.device!(params.device_number)
        t_start = time()
        println("SCALING LP ...")
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))

        CUDA.synchronize()
        t_start = time()
        println("COPY TO GPU ...")
        standard_lp_gpu = LP_info_gpu(CuSparseMatrixCSR(standard_lp.A),
            CuSparseMatrixCSR(standard_lp.A'),
            CuVector(standard_lp.c),
            CuVector(standard_lp.AL),
            CuVector(standard_lp.AU),
            CuVector(standard_lp.l),
            CuVector(standard_lp.u),
            standard_lp.obj_constant,
        )
        scaling_info_gpu = Scaling_info_gpu(CuVector(scaling_info.l_org),
            CuVector(scaling_info.u_org),
            CuVector(scaling_info.row_norm),
            CuVector(scaling_info.col_norm),
            scaling_info.b_scale,
            scaling_info.c_scale,
            scaling_info.norm_b,
            scaling_info.norm_c,
            scaling_info.norm_b_org,
            scaling_info.norm_c_org)
        CUDA.synchronize()
        println(@sprintf("COPY TO GPU time: %.2f seconds", time() - t_start))
    else
        t_start = time()
        println("SCALING LP ...")
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
    end

    setup_time = time() - setup_start
    if params.use_gpu
        results = solve(standard_lp_gpu, scaling_info_gpu, params)
    else
        results = solve(standard_lp, scaling_info, params)
    end
    println(@sprintf("Total time: %.2fs", read_time + setup_time + results.time),
        @sprintf("  read time = %.2fs", read_time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))

    return results
end


# it's used in demo_Abc.jl
function run_lp(A::SparseMatrixCSC,
    AL::Vector{Float64},
    AU::Vector{Float64},
    c::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64,
    params::HPRLP_parameters)

    if params.warm_up
        println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
        t_start_all = time()
        max_iter = params.max_iter
        params.max_iter = 200
        results = run_Abc(A, c, AL, AU, l, u, obj_constant, params)
        params.max_iter = max_iter
        all_time = time() - t_start_all
        println("warm up time: ", all_time)
        println("warm up ends ----------------------------------------------------------------------------------------------------------")
    end
    println("main run starts: ----------------------------------------------------------------------------------------------------------")
    results = run_Abc(A, c, AL, AU, l, u, obj_constant, params)
    println("main run ends----------------------------------------------------------------------------------------------------------")
    return results
end

# the function to run the HPR-LP algorithm on a single LP problem with A, b, c, l, u, m1, obj_constant
function run_Abc(A::SparseMatrixCSC,
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64,
    params::HPRLP_parameters)

    A = copy(A)
    c = copy(c)
    AL = copy(AL)
    AU = copy(AU)
    l = copy(l)
    u = copy(u)
    setup_start = time()
    standard_lp = LP_info_cpu(A, transpose(A), c, AL, AU, l, u, obj_constant)
    if params.use_gpu
        CUDA.device!(params.device_number)
        t_start = time()
        println("SCALING LP ...")
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))

        CUDA.synchronize()
        t_start = time()
        println("COPY TO GPU ...")
        standard_lp_gpu = LP_info_gpu(CuSparseMatrixCSR(standard_lp.A),
            CuSparseMatrixCSR(standard_lp.A'),
            CuVector(standard_lp.c),
            CuVector(standard_lp.AL),
            CuVector(standard_lp.AU),
            CuVector(standard_lp.l),
            CuVector(standard_lp.u),
            standard_lp.obj_constant,
        )
        scaling_info_gpu = Scaling_info_gpu(CuVector(scaling_info.l_org),
            CuVector(scaling_info.u_org),
            CuVector(scaling_info.row_norm),
            CuVector(scaling_info.col_norm),
            scaling_info.b_scale,
            scaling_info.c_scale,
            scaling_info.norm_b,
            scaling_info.norm_c,
            scaling_info.norm_b_org,
            scaling_info.norm_c_org)
        CUDA.synchronize()
        println(@sprintf("COPY TO GPU time: %.2f seconds", time() - t_start))
    else
        t_start = time()
        println("SCALING LP ...")
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
    end
    setup_time = time() - setup_start

    if params.use_gpu
        results = solve(standard_lp_gpu, scaling_info_gpu, params)
    else
        results = solve(standard_lp, scaling_info, params)
    end
    println(@sprintf("Total time: %.2fs", setup_time + results.time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))
    return results
end

# the function to test the HPR-LP algorithm on a dataset
function run_dataset(data_path::String, result_path::String, params::HPRLP_parameters)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = result_path * "HPRLP_result.csv"

    # redirect the output to a file
    log_path = result_path * "HPRLP_log.txt"

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        timelist = Vector{Any}(result_table.alg_time[1:end-2])
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time4list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time4list = []
        iter6list = []
        time6list = []
    end


    warm_up_done = false
    for i = 1:length(files)
        file = files[i]
        if file in namelist
            println("The result of problem exists: ", file)
        end
        if occursin(".mps", file) && !(file in namelist)
            FILE_NAME = data_path * file
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
            # println(file)

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                if params.warm_up && !warm_up_done
                    warm_up_done = true
                    println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                    t_start_all = time()
                    max_iter = params.max_iter
                    params.max_iter = 200
                    results = run_file(FILE_NAME, params)
                    params.max_iter = max_iter
                    all_time = time() - t_start_all
                    println("warm up time: ", all_time)
                    println("warm up ends ----------------------------------------------------------------------------------------------------------")
                end


                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, params.time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.output_type)
                push!(iter4list, results.iter_4)
                push!(time4list, min(results.time_4, params.time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, params.time_limit))
            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                alg_time=timelist,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_4=iter4list,
                time_4=time4list,
                iter_6=iter6list,
                time_6=time6list,
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_4 = exp(mean(log.(time4list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_4 = exp(mean(log.(iter4list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6])
            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, timelist)
            solved_4 = count(x -> x < params.time_limit, time4list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_4, "", solved_6])

            CSV.write(csv_file, result_table)
        end
    end
    println("The solver has finished running the dataset, total ", length(files), " problems")

    close(io)
end

# the function to test the HPR-LP algorithm on a single file
function run_single(file_name::String, params::HPRLP_parameters)

    println("data path: ", file_name)

    if occursin(".mps", file_name)
        if params.warm_up
            println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
            t_start_all = time()
            max_iter = params.max_iter
            params.max_iter = 200
            results = run_file(file_name, params)
            params.max_iter = max_iter
            all_time = time() - t_start_all
            println("warm up time: ", all_time)
            println("warm up ends ----------------------------------------------------------------------------------------------------------")
        end

        println("main run starts: ----------------------------------------------------------------------------------------------------------")
        results = run_file(file_name, params)
        println("main run ends----------------------------------------------------------------------------------------------------------")
    else
        error("The file is not in the correct format, please provide a .mps file")
    end

    return results
end
