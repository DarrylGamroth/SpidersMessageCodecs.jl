# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

export SparseMatrixCSXMessage, SparseMatrixCSXMessageDecoder, SparseMatrixCSXMessageEncoder
abstract type SparseMatrixCSXMessage{T} end

struct SparseMatrixCSXMessageDecoder{T<:AbstractArray{UInt8}} <: SparseMatrixCSXMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::UInt16
    acting_version::UInt16
    function SparseMatrixCSXMessageDecoder(buffer::T, offset::Integer, position_ptr::Ref{Int64},
        acting_block_length::Integer, acting_version::Integer) where {T}
        position_ptr[] = offset + acting_block_length
        new{T}(buffer, offset, position_ptr, acting_block_length, acting_version)
    end
end

struct SparseMatrixCSXMessageEncoder{T<:AbstractArray{UInt8},HasSbeHeader} <: SparseMatrixCSXMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    function SparseMatrixCSXMessageEncoder(buffer::T, offset::Integer,
        position_ptr::Ref{Int64}, hasSbeHeader::Bool=false) where {T}
        position_ptr[] = offset + 84
        new{T,hasSbeHeader}(buffer, offset, position_ptr)
    end
end

@inline function SparseMatrixCSXMessageDecoder(buffer::AbstractArray, offset::Integer=0;
    position_ptr::Base.RefValue{Int64}=Ref(0),
    header::MessageHeader=MessageHeader(buffer, offset))
    if templateId(header) != UInt16(0xb) || schemaId(header) != UInt16(0x1)
        throw(DomainError("Template id or schema id mismatch"))
    end
    SparseMatrixCSXMessageDecoder(buffer, offset + sbe_encoded_length(header), position_ptr,
        blockLength(header), version(header))
end
@inline function SparseMatrixCSXMessageEncoder(buffer::AbstractArray, offset::Integer=0;
    position_ptr::Base.RefValue{Int64}=Ref(0),
    header::MessageHeader=MessageHeader(buffer, offset))
    blockLength!(header, UInt16(0x54))
    templateId!(header, UInt16(0xb))
    schemaId!(header, UInt16(0x1))
    version!(header, UInt16(0x0))
    SparseMatrixCSXMessageEncoder(buffer, offset + sbe_encoded_length(header), position_ptr, true)
end
sbe_buffer(m::SparseMatrixCSXMessage) = m.buffer
sbe_offset(m::SparseMatrixCSXMessage) = m.offset
sbe_position_ptr(m::SparseMatrixCSXMessage) = m.position_ptr
sbe_position(m::SparseMatrixCSXMessage) = m.position_ptr[]
sbe_position!(m::SparseMatrixCSXMessage, position) = m.position_ptr[] = position
sbe_block_length(::SparseMatrixCSXMessage) = UInt16(0x54)
sbe_block_length(::Type{<:SparseMatrixCSXMessage}) = UInt16(0x54)
sbe_template_id(::SparseMatrixCSXMessage) = UInt16(0xb)
sbe_template_id(::Type{<:SparseMatrixCSXMessage})  = UInt16(0xb)
sbe_schema_id(::SparseMatrixCSXMessage) = UInt16(0x1)
sbe_schema_id(::Type{<:SparseMatrixCSXMessage})  = UInt16(0x1)
sbe_schema_version(::SparseMatrixCSXMessage) = UInt16(0x0)
sbe_schema_version(::Type{<:SparseMatrixCSXMessage})  = UInt16(0x0)
sbe_semantic_type(::SparseMatrixCSXMessage) = ""
sbe_semantic_version(::SparseMatrixCSXMessage) = ""
sbe_acting_block_length(m::SparseMatrixCSXMessageDecoder) = m.acting_block_length
sbe_acting_block_length(::SparseMatrixCSXMessageEncoder) = UInt16(0x54)
sbe_acting_version(m::SparseMatrixCSXMessageDecoder) = m.acting_version
sbe_acting_version(::SparseMatrixCSXMessageEncoder) = UInt16(0x0)
sbe_rewind!(m::SparseMatrixCSXMessage) = sbe_position!(m, m.offset + sbe_acting_block_length(m))
sbe_encoded_length(m::SparseMatrixCSXMessage) = sbe_position(m) - m.offset
@inline function sbe_decoded_length(m::SparseMatrixCSXMessage)
    skipper = SparseMatrixCSXMessageDecoder(sbe_buffer(m), sbe_offset(m), Ref(0),
        sbe_acting_block_length(m), sbe_acting_version(m))
    sbe_skip!(skipper)
    sbe_encoded_length(skipper)
end

Base.sizeof(m::SparseMatrixCSXMessage) = sbe_decoded_length(m)
function Base.convert(::Type{AbstractArray{UInt8}}, m::SparseMatrixCSXMessageEncoder{<:AbstractArray{UInt8},true})
    return view(m.buffer, m.offset+1-sbe_encoded_length(MessageHeader):m.offset+sbe_encoded_length(m))
end
function Base.convert(::Type{AbstractArray{UInt8}}, m::SparseMatrixCSXMessageEncoder{<:AbstractArray{UInt8},false})
    return view(m.buffer, m.offset+1:m.offset+sbe_encoded_length(m))
end

function header_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
header_id(::SparseMatrixCSXMessage) = UInt16(0x1)
header_since_version(::SparseMatrixCSXMessage) = UInt16(0x0)
header_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= UInt16(0x0)
header_encoding_offset(::SparseMatrixCSXMessage) = 0
header(m::SparseMatrixCSXMessage) = SpidersMessageHeader(m.buffer, m.offset + 0, sbe_acting_version(m))

function format_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
format_id(::SparseMatrixCSXMessage) = UInt16(0x2)
format_since_version(::SparseMatrixCSXMessage) = UInt16(0x0)
format_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= UInt16(0x0)
format_encoding_offset(::SparseMatrixCSXMessage) = 64
format_encoding_length(::SparseMatrixCSXMessage) = 1
@inline function format(m::SparseMatrixCSXMessageDecoder, ::Type{Integer})
    return decode_le(Int8, m.buffer, m.offset + 64)
end
@inline function format(m::SparseMatrixCSXMessageDecoder)
    return Format.SbeEnum(decode_le(Int8, m.buffer, m.offset + 64))
end
@inline format!(m::SparseMatrixCSXMessageEncoder, value::Format.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 64, Int8(value))

function majorOrder_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
majorOrder_id(::SparseMatrixCSXMessage) = UInt16(0x3)
majorOrder_since_version(::SparseMatrixCSXMessage) = UInt16(0x0)
majorOrder_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= UInt16(0x0)
majorOrder_encoding_offset(::SparseMatrixCSXMessage) = 65
majorOrder_encoding_length(::SparseMatrixCSXMessage) = 1
@inline function majorOrder(m::SparseMatrixCSXMessageDecoder, ::Type{Integer})
    return decode_le(Int8, m.buffer, m.offset + 65)
end
@inline function majorOrder(m::SparseMatrixCSXMessageDecoder)
    return MajorOrder.SbeEnum(decode_le(Int8, m.buffer, m.offset + 65))
end
@inline majorOrder!(m::SparseMatrixCSXMessageEncoder, value::MajorOrder.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 65, Int8(value))

function indexing_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
indexing_id(::SparseMatrixCSXMessage) = UInt16(0x4)
indexing_since_version(::SparseMatrixCSXMessage) = UInt16(0x0)
indexing_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= UInt16(0x0)
indexing_encoding_offset(::SparseMatrixCSXMessage) = 66
indexing_encoding_length(::SparseMatrixCSXMessage) = 1
@inline function indexing(m::SparseMatrixCSXMessageDecoder, ::Type{Integer})
    return decode_le(Int8, m.buffer, m.offset + 66)
end
@inline function indexing(m::SparseMatrixCSXMessageDecoder)
    return Indexing.SbeEnum(decode_le(Int8, m.buffer, m.offset + 66))
end
@inline indexing!(m::SparseMatrixCSXMessageEncoder, value::Indexing.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 66, Int8(value))

function reserved1_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
reserved1_id(::SparseMatrixCSXMessage) = UInt16(0x5)
reserved1_since_version(::SparseMatrixCSXMessage) = UInt16(0x0)
reserved1_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= UInt16(0x0)
reserved1_encoding_offset(::SparseMatrixCSXMessage) = 67
reserved1_null_value(::SparseMatrixCSXMessage) = Int8(-128)
reserved1_min_value(::SparseMatrixCSXMessage) = Int8(-127)
reserved1_max_value(::SparseMatrixCSXMessage) = Int8(127)
reserved1_encoding_length(::SparseMatrixCSXMessage) = 1

@inline function reserved1(m::SparseMatrixCSXMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 67)
end
@inline reserved1!(m::SparseMatrixCSXMessageEncoder, value) = encode_le(Int8, m.buffer, m.offset + 67, value)

function dims_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
dims_id(::SparseMatrixCSXMessage) = UInt16(0x6)
dims_since_version(::SparseMatrixCSXMessage) = UInt16(0x0)
dims_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= UInt16(0x0)
dims_encoding_offset(::SparseMatrixCSXMessage) = 68
dims_null_value(::SparseMatrixCSXMessage) = Int64(-9223372036854775808)
dims_min_value(::SparseMatrixCSXMessage) = Int64(-9223372036854775807)
dims_max_value(::SparseMatrixCSXMessage) = Int64(9223372036854775807)
dims_encoding_length(::SparseMatrixCSXMessage) = 16
dims_length(::SparseMatrixCSXMessage) = 2
dims_eltype(::SparseMatrixCSXMessage) = Int64

@inline function dims(m::SparseMatrixCSXMessageDecoder)
    return mappedarray(ltoh, reinterpret(Int64, view(m.buffer, m.offset+68+1:m.offset+68+sizeof(Int64)*2)))
end

@inline function dims(m::SparseMatrixCSXMessageDecoder, ::Type{<:SVector})
    return mappedarray(ltoh, reinterpret(SVector{2,Int64}, view(m.buffer, m.offset+68+1:m.offset+68+sizeof(Int64)*2))[])
end

@inline function dims!(m::SparseMatrixCSXMessageEncoder)
    return mappedarray(ltoh, htol, reinterpret(Int64, view(m.buffer, m.offset+68+1:m.offset+68+sizeof(Int64)*2)))
end

@inline function dims!(m::SparseMatrixCSXMessageEncoder, value)
    copyto!(dims!(m), value)
end

function indexPointer_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

indexPointer_character_encoding(::SparseMatrixCSXMessage) = "null"
indexPointer_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= 0
indexPointer_id(::SparseMatrixCSXMessage) = 20
indexPointer_header_length(::SparseMatrixCSXMessage) = 4

@inline function indexPointer_length(m::SparseMatrixCSXMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function indexPointer_length!(m::SparseMatrixCSXMessageEncoder, n)
    @boundscheck n > 1073741824 && throw(ArgumentError("length exceeds schema limit"))
    @boundscheck checkbounds(m.buffer, sbe_position(m) + 4 + n)
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_indexPointer!(m::SparseMatrixCSXMessageDecoder)
    len = indexPointer_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return len
end

@inline function indexPointer(m::SparseMatrixCSXMessageDecoder)
    len = indexPointer_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline indexPointer(m::SparseMatrixCSXMessageDecoder, ::Type{AbstractArray{T}}) where {T<:Real} = reinterpret(T, indexPointer(m))
@inline indexPointer(m::SparseMatrixCSXMessageDecoder, ::Type{NTuple{N,T}}) where {N,T<:Real} = (x = reinterpret(T, indexPointer(m)); ntuple(i -> x[i], Val(N)))
@inline indexPointer(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:AbstractString} = StringView(rstrip_nul(indexPointer(m)))
@inline indexPointer(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Symbol} = Symbol(indexPointer(m, StringView))
@inline indexPointer(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Real} = reinterpret(T, indexPointer(m))[]
@inline indexPointer(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Nothing} = (skip_indexPointer!(m); nothing)

@inline function indexPointer_buffer!(m::SparseMatrixCSXMessageEncoder, len)
    indexPointer_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function indexPointer!(m::SparseMatrixCSXMessageEncoder, src::AbstractArray)
    len = sizeof(src)
    indexPointer_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(UInt8, src))
end

@inline function indexPointer!(m::SparseMatrixCSXMessageEncoder, src::NTuple)
    len = sizeof(src)
    indexPointer_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(NTuple{len,UInt8}, src))
end

@inline function indexPointer!(m::SparseMatrixCSXMessageEncoder, src::AbstractString)
    len = sizeof(src)
    indexPointer_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, codeunits(src))
end

@inline indexPointer!(m::SparseMatrixCSXMessageEncoder, src::Symbol) = indexPointer!(m, to_string(src))
@inline indexPointer!(m::SparseMatrixCSXMessageEncoder, src::StaticString) = indexPointer!(m, Tuple(src))
@inline indexPointer!(m::SparseMatrixCSXMessageEncoder, src::Real) = indexPointer!(m, Tuple(src))
@inline indexPointer!(m::SparseMatrixCSXMessageEncoder, ::Nothing) = indexPointer_buffer!(m, 0)

function indicies_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

indicies_character_encoding(::SparseMatrixCSXMessage) = "null"
indicies_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= 0
indicies_id(::SparseMatrixCSXMessage) = 21
indicies_header_length(::SparseMatrixCSXMessage) = 4

@inline function indicies_length(m::SparseMatrixCSXMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function indicies_length!(m::SparseMatrixCSXMessageEncoder, n)
    @boundscheck n > 1073741824 && throw(ArgumentError("length exceeds schema limit"))
    @boundscheck checkbounds(m.buffer, sbe_position(m) + 4 + n)
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_indicies!(m::SparseMatrixCSXMessageDecoder)
    len = indicies_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return len
end

@inline function indicies(m::SparseMatrixCSXMessageDecoder)
    len = indicies_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline indicies(m::SparseMatrixCSXMessageDecoder, ::Type{AbstractArray{T}}) where {T<:Real} = reinterpret(T, indicies(m))
@inline indicies(m::SparseMatrixCSXMessageDecoder, ::Type{NTuple{N,T}}) where {N,T<:Real} = (x = reinterpret(T, indicies(m)); ntuple(i -> x[i], Val(N)))
@inline indicies(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:AbstractString} = StringView(rstrip_nul(indicies(m)))
@inline indicies(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Symbol} = Symbol(indicies(m, StringView))
@inline indicies(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Real} = reinterpret(T, indicies(m))[]
@inline indicies(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Nothing} = (skip_indicies!(m); nothing)

@inline function indicies_buffer!(m::SparseMatrixCSXMessageEncoder, len)
    indicies_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function indicies!(m::SparseMatrixCSXMessageEncoder, src::AbstractArray)
    len = sizeof(src)
    indicies_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(UInt8, src))
end

@inline function indicies!(m::SparseMatrixCSXMessageEncoder, src::NTuple)
    len = sizeof(src)
    indicies_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(NTuple{len,UInt8}, src))
end

@inline function indicies!(m::SparseMatrixCSXMessageEncoder, src::AbstractString)
    len = sizeof(src)
    indicies_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, codeunits(src))
end

@inline indicies!(m::SparseMatrixCSXMessageEncoder, src::Symbol) = indicies!(m, to_string(src))
@inline indicies!(m::SparseMatrixCSXMessageEncoder, src::StaticString) = indicies!(m, Tuple(src))
@inline indicies!(m::SparseMatrixCSXMessageEncoder, src::Real) = indicies!(m, Tuple(src))
@inline indicies!(m::SparseMatrixCSXMessageEncoder, ::Nothing) = indicies_buffer!(m, 0)

function values_meta_attribute(::SparseMatrixCSXMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

values_character_encoding(::SparseMatrixCSXMessage) = "null"
values_in_acting_version(m::SparseMatrixCSXMessage) = sbe_acting_version(m) >= 0
values_id(::SparseMatrixCSXMessage) = 30
values_header_length(::SparseMatrixCSXMessage) = 4

@inline function values_length(m::SparseMatrixCSXMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function values_length!(m::SparseMatrixCSXMessageEncoder, n)
    @boundscheck n > 1073741824 && throw(ArgumentError("length exceeds schema limit"))
    @boundscheck checkbounds(m.buffer, sbe_position(m) + 4 + n)
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_values!(m::SparseMatrixCSXMessageDecoder)
    len = values_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return len
end

@inline function values(m::SparseMatrixCSXMessageDecoder)
    len = values_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline values(m::SparseMatrixCSXMessageDecoder, ::Type{AbstractArray{T}}) where {T<:Real} = reinterpret(T, values(m))
@inline values(m::SparseMatrixCSXMessageDecoder, ::Type{NTuple{N,T}}) where {N,T<:Real} = (x = reinterpret(T, values(m)); ntuple(i -> x[i], Val(N)))
@inline values(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:AbstractString} = StringView(rstrip_nul(values(m)))
@inline values(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Symbol} = Symbol(values(m, StringView))
@inline values(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Real} = reinterpret(T, values(m))[]
@inline values(m::SparseMatrixCSXMessageDecoder, ::Type{T}) where {T<:Nothing} = (skip_values!(m); nothing)

@inline function values_buffer!(m::SparseMatrixCSXMessageEncoder, len)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::SparseMatrixCSXMessageEncoder, src::AbstractArray)
    len = sizeof(src)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(UInt8, src))
end

@inline function values!(m::SparseMatrixCSXMessageEncoder, src::NTuple)
    len = sizeof(src)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(NTuple{len,UInt8}, src))
end

@inline function values!(m::SparseMatrixCSXMessageEncoder, src::AbstractString)
    len = sizeof(src)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, codeunits(src))
end

@inline values!(m::SparseMatrixCSXMessageEncoder, src::Symbol) = values!(m, to_string(src))
@inline values!(m::SparseMatrixCSXMessageEncoder, src::StaticString) = values!(m, Tuple(src))
@inline values!(m::SparseMatrixCSXMessageEncoder, src::Real) = values!(m, Tuple(src))
@inline values!(m::SparseMatrixCSXMessageEncoder, ::Nothing) = values_buffer!(m, 0)

function show(io::IO, m::SparseMatrixCSXMessage{T}) where {T}
    println(io, "SparseMatrixCSXMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = SparseMatrixCSXMessageDecoder(sbe_buffer(m), sbe_offset(m), Ref(0),
        sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    show(io, header(writer))

    println(io)
    print(io, "format: ")
    print(io, format(writer))

    println(io)
    print(io, "majorOrder: ")
    print(io, majorOrder(writer))

    println(io)
    print(io, "indexing: ")
    print(io, indexing(writer))

    println(io)
    print(io, "reserved1: ")
    print(io, reserved1(writer))

    println(io)
    print(io, "dims: ")
    print(io, dims(writer))

    println(io)
    print(io, "indexPointer: ")
    print(io, skip_indexPointer!(writer))
    print(io, " bytes of raw data")

    println(io)
    print(io, "indicies: ")
    print(io, skip_indicies!(writer))
    print(io, " bytes of raw data")

    println(io)
    print(io, "values: ")
    print(io, skip_values!(writer))
    print(io, " bytes of raw data")

    nothing
end

@inline function sbe_skip!(m::SparseMatrixCSXMessageDecoder)
    sbe_rewind!(m)
    skip_indexPointer!(m)
    skip_indicies!(m)
    skip_values!(m)
    return
end
