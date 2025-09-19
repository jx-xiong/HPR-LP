module HPRLP

using SparseArrays
using LinearAlgebra
using QPSReader
using CUDA
using CUDA.CUSPARSE
using Printf
using CSV
using DataFrames
using Random
using Statistics
using Logging

FloatType = Float32
println("FloatType = ", FloatType)

include("structs.jl")
include("utils.jl")
include("kernels.jl")
include("algorithm.jl")

end