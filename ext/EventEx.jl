module EventEx

using ..Event
using ..Sbe
using StringViews
using UnsafeArrays

# Return the SBE message type for the given templateId and schemaId
function Sbe.message_type(
    ::Val{Event.sbe_template_id(Event.EventMessage)},
    ::Val{Event.sbe_schema_id(Event.EventMessage)})
    Event.EventMessage
end

function Sbe.decoder(::Type{<:Event.EventMessage}, buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue)
    Event.EventMessageDecoder(buffer, offset, position_ptr, Event.MessageHeader(buffer, offset))
end

function Sbe.sbe_decoded_length(m::Event.EventMessage)
    Event.sbe_decoded_length(m)
end

Sbe.is_sbe_message(::Type{<:Event.EventMessage}) = true

function Base.convert(::Type{<:AbstractArray{UInt8}}, m::Event.EventMessage)
    offset = Event.sbe_offset(m) - Event.sbe_encoded_length(Event.MessageHeader)
    offset < 0 && throw(ArgumentError("Message offset is negative"))
    len = Event.sbe_decoded_length(m)
    return view(Event.sbe_buffer(m), offset+1:Event.sbe_offset(m)+len)
end

abstract type SbeType end

function Base.eltype(T::Event.Format.SbeEnum)
    T == Event.Format.NOTHING ? Nothing :
    T == Event.Format.UINT8 ? UInt8 :
    T == Event.Format.INT8 ? Int8 :
    T == Event.Format.UINT16 ? UInt16 :
    T == Event.Format.INT16 ? Int16 :
    T == Event.Format.UINT32 ? UInt32 :
    T == Event.Format.INT32 ? Int32 :
    T == Event.Format.UINT64 ? UInt64 :
    T == Event.Format.INT64 ? Int64 :
    T == Event.Format.FLOAT32 ? Float32 :
    T == Event.Format.FLOAT64 ? Float64 :
    T == Event.Format.BOOLEAN ? Bool :
    T == Event.Format.STRING ? AbstractString :
    T == Event.Format.BYTES ? AbstractVector{UInt8} :
    T == Event.Format.SBE ? SbeType :
    throw(ArgumentError("unexpected format"))
end

function Base.convert(::Type{Event.Format.SbeEnum}, ::Type{T}) where {T}
    T === Nothing ? Event.Format.NOTHING :
    T === UInt8 ? Event.Format.UINT8 :
    T === Int8 ? Event.Format.INT8 :
    T === UInt16 ? Event.Format.UINT16 :
    T === Int16 ? Event.Format.INT16 :
    T === UInt32 ? Event.Format.UINT32 :
    T === Int32 ? Event.Format.INT32 :
    T === UInt64 ? Event.Format.UINT64 :
    T === Int64 ? Event.Format.INT64 :
    T === Float32 ? Event.Format.FLOAT32 :
    T === Float64 ? Event.Format.FLOAT64 :
    T === Bool ? Event.Format.BOOLEAN :
    T <: AbstractString ? Event.Format.STRING :
    T <: AbstractVector{UInt8} ? Event.Format.BYTES :
    T === Symbol ? Event.Format.STRING :
    T <: SbeType ? Event.Format.SBE :
    Sbe.is_sbe_message(T) ? Event.Format.SBE :
    throw(ArgumentError("unexpected type"))
end

Event.key(::Type{Symbol}, m::Event.EventMessage) = Symbol(Event.key(String, m))
Event.value(::Type{Nothing}, m::Event.EventMessage) = nothing
Event.value(::Type{Symbol}, m::Event.EventMessage) = Symbol(Event.value(String, m))
Event.value(::Type{T}, m::Event.EventMessage) where {T<:Real} = reinterpret(T, Event.value(m))[]
Event.value(::Type{<:AbstractString}, m::Event.EventMessage) = StringView(Event.value(m))
Event.value(::Type{<:AbstractVector{UInt8}}, m::Event.EventMessage) = Event.value(m)
Event.value(::Type{<:SbeType}, m::Event.EventMessage) = Sbe.decoder(Event.value(m), m.position_ptr)
Event.value!(::Type{T}, m::Event.EventMessageEncoder, len::Int) where {T<:Real} = reinterpret(T, Event.value!(m, sizeof(T) * len))
Event.value!(m::Event.EventMessageEncoder, value::Symbol) = Event.value!(m, convert(UnsafeArray{UInt8}, value))
# Event.value!(m::Event.EventMessage, key_value::Tuple{AbstractString,T}) where {T} = Event.value!(m, key_value...)

@inline function Event.value(::Type{T}, m::Event.EventMessage) where {T}
    Sbe.is_sbe_message(T) || throw(ArgumentError("unexpected type, got $T"))
    Sbe.decoder(T, Event.value(m), m.position_ptr)
end

@inline function (m::Event.EventMessageDecoder)()
    Event.sbe_rewind!(m)
    T = Base.eltype(Event.format(m))
    Pair(Event.key(Symbol, m), Event.value(T, m))
end

@inline function (m::Event.EventMessageDecoder)(::Type{T}) where {T}
    Event.sbe_rewind!(m)
    Pair(Event.key(Symbol, m), Event.value(T, m))
end

@inline function key!(m::Event.EventMessage, key)
    buf = Event.key!(m)
    fill!(buf, 0)
    copyto!(buf, key)
end

@inline function (m::Event.EventMessageEncoder)(key::AbstractString, ::Nothing)
    Event.sbe_rewind!(m)
    Event.format!(m, convert(Event.Format.SbeEnum, Nothing))
    key!(m, key)
    Event.skip_value!(m)
    nothing
end

@inline function (m::Event.EventMessageEncoder)(key::AbstractString, value::T) where {T<:Real}
    Event.sbe_rewind!(m)
    Event.format!(m, convert(Event.Format.SbeEnum, T))
    key!(m, key)
    Event.value!(T, m, 1)[] = value
    nothing
end

@inline function (m::Event.EventMessageEncoder)(key::AbstractString, value::T) where {T<:Union{AbstractString, AbstractVector{UInt8}, Symbol}}
    Event.sbe_rewind!(m)
    Event.format!(m, convert(Event.Format.SbeEnum, T))
    key!(m, key)
    Event.value!(m, value)
    nothing
end

@inline function (m::Event.EventMessageEncoder)(key::AbstractString, value::T) where {T}
    Event.sbe_rewind!(m)
    Sbe.is_sbe_message(T) || throw(ArgumentError("$T is not an SBE message"))
    Event.format!(m, convert(Event.Format.SbeEnum, T))
    key!(m, key)
    Event.value!(m, convert(AbstractArray{UInt8}, value))
    nothing
end

function Base.show(io::IO, m::Event.EventMessage{T}) where {T}
    println(io, "EventMessage view over a type $T")
    println(io, "SbeBlockLength: ", Event.sbe_block_length(m))
    println(io, "SbeTemplateId:  ", Event.sbe_template_id(m))
    println(io, "SbeSchemaId:    ", Event.sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", Event.sbe_schema_version(m))

    writer = Event.EventMessageDecoder(Event.sbe_buffer(m), Event.sbe_offset(m),
        Event.sbe_block_length(m), Event.sbe_schema_version(m))
    print(io, "header: ")
    Event.show(io, Event.header(writer))

    println(io)
    print(io, "format: ")
    print(io, Event.format(writer))

    println(io)
    print(io, "key: ")
    print(io, writer().first)

    println(io)
    print(io, "value: ")
    print(io, writer().second)
end

end # module EventEx