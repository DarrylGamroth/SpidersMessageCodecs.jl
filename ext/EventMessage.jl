message_type(::Val{sbe_template_id(EventMessage)}) = EventMessage

function SpidersMessageCodecs.decode(::Type{<:EventMessage}, buffer::AbstractArray, args...; kwargs...)
    EventMessageDecoder(buffer::AbstractArray, args...; kwargs...)
end
function SpidersMessageCodecs.encode(::Type{<:EventMessage}, buffer::AbstractArray, args...; kwargs...)
    EventMessageEncoder(buffer::AbstractArray, args...; kwargs...)
end

is_sbe_message(::Type{<:SpidersMessageCodecs.EventMessage}) = true

@inline SpidersMessageCodecs.key(::Type{Symbol}, m::EventMessage) = Symbol(SpidersMessageCodecs.key(StringView, m))

@inline SpidersMessageCodecs.value(::Type{Nothing}, m::EventMessage) = nothing
@inline SpidersMessageCodecs.value(::Type{T}, m::EventMessage) where {T<:AbstractString} = StringView(SpidersMessageCodecs.value(m))
@inline SpidersMessageCodecs.value(::Type{T}, m::EventMessage) where {T<:AbstractVector{UInt8}} = SpidersMessageCodecs.value(m)
@inline SpidersMessageCodecs.value(::Type{T}, m::EventMessage) where {T<:Symbol} = Symbol(SpidersMessageCodecs.value(StringView, m))
@inline SpidersMessageCodecs.value(::Type{T}, m::EventMessage) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.value(m))[]
@inline SpidersMessageCodecs.value(::Type{T}, m::EventMessage) where {T<:SbeType} = SpidersMessageCodecs.decode(SpidersMessageCodecs.value(m); position_ptr=m.position_ptr)
@inline function SpidersMessageCodecs.value(::Type{T}, m::EventMessage) where {T}
    is_sbe_message(T) || throw(ArgumentError("unexpected type, got $T"))
    SpidersMessageCodecs.decode(T, SpidersMessageCodecs.value(m); position_ptr=m.position_ptr)
end

@inline SpidersMessageCodecs.value!(m::EventMessageEncoder, ::Nothing) = SpidersMessageCodecs.skip_value!(m)
@inline SpidersMessageCodecs.value!(m::EventMessageEncoder, value::Symbol) = SpidersMessageCodecs.value!(m, convert(StringView, value))
@inline SpidersMessageCodecs.value!(m::EventMessageEncoder, value::T) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.value!(m; length=sizeof(T)))[] = value

@inline function SpidersMessageCodecs.decode(m::EventMessageDecoder)
    T = Base.eltype(SpidersMessageCodecs.format(m))
    SpidersMessageCodecs.decode(T, m)
end

@inline function SpidersMessageCodecs.decode(::Type{T}, m::EventMessageDecoder) where {T}
    sbe_rewind!(m)
    Pair(SpidersMessageCodecs.key(Symbol, m), SpidersMessageCodecs.value(T, m))
end

@inline function SpidersMessageCodecs.encode(m::EventMessageEncoder, key::AbstractString, value::T) where {T<:Union{Nothing,AbstractString,AbstractVector{UInt8},<:Real,Symbol}}
    sbe_rewind!(m)
    SpidersMessageCodecs.format!(m, convert(SpidersMessageCodecs.Format.SbeEnum, T))
    SpidersMessageCodecs.key!(m, key)
    SpidersMessageCodecs.value!(m, value)
    nothing
end

@inline function SpidersMessageCodecs.encode(m::EventMessageEncoder, key::AbstractString, value::T) where {T}
    sbe_rewind!(m)
    is_sbe_message(T) || throw(ArgumentError("$T is not an SBE message"))
    SpidersMessageCodecs.format!(m, convert(SpidersMessageCodecs.Format.SbeEnum, T))
    SpidersMessageCodecs.key!(m, key)
    SpidersMessageCodecs.value!(m, convert(AbstractArray{UInt8}, value))
    nothing
end

@inline function SpidersMessageCodecs.encode(m::EventMessageEncoder, key::AbstractString, value::Enum)
    SpidersMessageCodecs.encode(m, key, Int64(value))
end

@inline function SpidersMessageCodecs.encode(m::EventMessageEncoder, key::Symbol, value)
    SpidersMessageCodecs.encode(m, convert(StringView, key), value)
end

@inline function SpidersMessageCodecs.encode(m::EventMessageEncoder, pair::Pair)
    SpidersMessageCodecs.encode(m, pair.first, pair.second)
end

function Base.show(io::IO, m::EventMessage{T}) where {T}
    println(io, "EventMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = EventMessageDecoder(sbe_buffer(m), sbe_offset(m), Ref(0),
        sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    SpidersMessageCodecs.show(io, SpidersMessageCodecs.header(writer))

    println(io)
    print(io, "format: ")
    print(io, SpidersMessageCodecs.format(writer))

    println(io)
    print(io, "key: ")
    print(io, SpidersMessageCodecs.decode(writer).first)

    println(io)
    print(io, "value: ")
    print(io, SpidersMessageCodecs.decode(writer).second)
end
