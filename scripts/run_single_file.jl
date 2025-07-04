import HPRLP

file_name = "xxx" # Replace with the actual path to your LP file

params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0
params.warm_up = false
params.use_gpu = true

result = HPRLP.run_single(file_name, params)

println("Objective value: ", result.primal_obj)