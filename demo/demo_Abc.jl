using SparseArrays
using LinearAlgebra
import HPRLP


# min c * x
# s.t. A_1 * x = b_1
#      A_2 * x >= b_2
#      l <= x <= u
# A_1 has m1 rows, A = [A_1; A_2] should be non-empty


# Example 1
# min -3x1 - 5x2
# s.t. -x1 - 2x2 >= -10
#      -3x1 - x2 >= -12
#      x1 >= 0, x2 >= 0
#      x1 <= Inf, x2 <= Inf

A = sparse([-1 -2; -3 -1])
b = Vector{Float64}([-10, -12])
c = Vector{Float64}([-3, -5])
l = Vector{Float64}([0, 0])
u = Vector{Float64}([Inf, Inf])

# the first m1 rows of A are equality constraints, the rest are ≥ constraints
m1 = 0
obj_constant = 0.0

params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0
params.use_gpu = true

result = HPRLP.run_lp(A, b, c, l, u, m1, obj_constant, params)

println("Objective value: ", result.primal_obj)
println("x1 = ", result.x[1])
println("x2 = ", result.x[2])
