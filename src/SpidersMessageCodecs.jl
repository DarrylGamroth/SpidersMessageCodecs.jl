# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

module SpidersMessageCodecs

using EnumX
using MappedArrays
using StaticArrays
using StaticStrings
using StringViews
using UnsafeArrays

include("Utils.jl")
include("Format.jl")
include("MajorOrder.jl")
include("Indexing.jl")
include("SpidersMessageHeader.jl")
include("MessageHeader.jl")
include("GroupSizeEncoding.jl")
include("VarStringEncoding.jl")
include("VarDataEncoding.jl")
include("EventMessage.jl")
include("TensorMessage.jl")
include("SparseMatrixCSXMessage.jl")
include("SparseVectorMessage.jl")
include("DiagonalMatrixMessage.jl")
include("TensorStreamHeaderMessage.jl")
include("ChunkHeaderMessage.jl")
include("TensorStreamDataMessage.jl")
include("ChunkDataMessage.jl")

export sbe_buffer,
    sbe_offset,
    sbe_position_ptr,
    sbe_position,
    sbe_position!,
    sbe_block_length,
    sbe_template_id,
    sbe_schema_id,
    sbe_schema_version,
    sbe_semantic_type,
    sbe_semantic_version,
    sbe_acting_block_length,
    sbe_acting_version,
    sbe_rewind!,
    sbe_encoded_length,
    sbe_decoded_length,
    sbe_skip!,
    decode,
    encode

function decode end
function encode end
end
