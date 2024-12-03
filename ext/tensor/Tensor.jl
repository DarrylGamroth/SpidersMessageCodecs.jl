# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

module Tensor

using EnumX
using MappedArrays
using StaticArrays
using StringViews

include("Utils.jl")
include("Order.jl")
include("Format.jl")
include("Indexing.jl")
include("SpidersMessageHeader.jl")
include("MessageHeader.jl")
include("VarDataEncoding.jl")
include("TensorMessage.jl")
include("SparseMatrixCSXMessage.jl")
include("SparseVectorMessage.jl")
include("DiagonalMatrixMessage.jl")
end