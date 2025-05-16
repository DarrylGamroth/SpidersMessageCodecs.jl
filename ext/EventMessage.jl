message_type(::Val{sbe_template_id(EventMessage)}) = EventMessage

@inline function SpidersMessageCodecs.decode(::Type{<:EventMessage}, buffer::AbstractArray, args...; kwargs...)
    EventMessageDecoder(buffer::AbstractArray, args...; kwargs...)
end
@inline function SpidersMessageCodecs.encode(::Type{<:EventMessage}, buffer::AbstractArray, args...; kwargs...)
    EventMessageEncoder(buffer::AbstractArray, args...; kwargs...)
end

Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{<:SpidersMessageCodecs.EventMessage}) = SpidersMessageCodecs.Format.SBE

@inline SpidersMessageCodecs.value(m::EventMessageDecoder, ::Type{T}) where {T<:SbeType} = SpidersMessageCodecs.decode(SpidersMessageCodecs.value(m); position_ptr=m.position_ptr)
@inline SpidersMessageCodecs.value(m::EventMessageDecoder, ::Type{T}) where {T} = SpidersMessageCodecs.decode(T, SpidersMessageCodecs.value(m); position_ptr=m.position_ptr)
@inline SpidersMessageCodecs.value!(m::EventMessageEncoder, value::T) where {T<:Enum} = SpidersMessageCodecs.value!(m, Int64(value))
@inline SpidersMessageCodecs.value!(m::EventMessageEncoder, value) = SpidersMessageCodecs.value!(m, Base.convert(AbstractArray{UInt8}, value))

@inline function SpidersMessageCodecs.decode(m::EventMessageDecoder, ::Type{T}) where {T}
    sbe_rewind!(m)
    SpidersMessageCodecs.value(m, T)
end

@inline function SpidersMessageCodecs.decode(m::EventMessageDecoder)
    T = Base.eltype(SpidersMessageCodecs.format(m))
    SpidersMessageCodecs.decode(m, T)
end

@inline function SpidersMessageCodecs.encode(m::EventMessageEncoder, value::T) where {T}
    sbe_rewind!(m)
    SpidersMessageCodecs.format!(m, convert(SpidersMessageCodecs.Format.SbeEnum, T))
    SpidersMessageCodecs.value!(m, value)
    nothing
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
    print(io, SpidersMessageCodecs.key(writer, String))

    println(io)
    print(io, "value: ")
    print(io, SpidersMessageCodecs.decode(writer))
end
