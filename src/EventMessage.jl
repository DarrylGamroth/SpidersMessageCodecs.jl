include("../gen/event/Event.jl")

# FIXME Add Event key-value-type to the SBE schema using Tuple or NamedTuple
# Add abstract type for all SBE messages so the message type can passed to the Tuple or NamedTuple
# or retrieve

# Should add an abstract type for all SBE messages so the message type can be passed
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
    T == Event.Format.SBE ? AbstractVector{UInt8} :
    throw(ArgumentError("unexpected format"))
end

function event_eltype(::Type{T}) where {T}
    T == Nothing ? Event.Format.NOTHING :
    T == UInt8 ? Event.Format.UINT8 :
    T == Int8 ? Event.Format.INT8 :
    T == UInt16 ? Event.Format.UINT16 :
    T == Int16 ? Event.Format.INT16 :
    T == UInt32 ? Event.Format.UINT32 :
    T == Int32 ? Event.Format.INT32 :
    T == UInt64 ? Event.Format.UINT64 :
    T == Int64 ? Event.Format.INT64 :
    T == Float32 ? Event.Format.FLOAT32 :
    T == Float64 ? Event.Format.FLOAT64 :
    T == Bool ? Event.Format.BOOLEAN :
    T <: AbstractString ? Event.Format.STRING :
    T <: AbstractVector{UInt8} ? Event.Format.BYTES :
    throw(ArgumentError("unexpected type"))
end

Event.value(::Type{Nothing}, m::Event.EventMessage) = nothing
Event.value(::Type{AbstractString}, m::Event.EventMessage) = StringView(Event.value(m))
Event.value(::Type{T}, m::Event.EventMessage) where {T<:Real} = reinterpret(T, Event.value(m))[]
Event.value(::Type{AbstractVector{UInt8}}, m::Event.EventMessage) = Event.value(m)

function Event.value(::Type{Tuple}, m::Event.EventMessage)
    Event.sbe_rewind!(m)
    type = Base.eltype(Event.format(m))
    Event.value(Tuple{type}, m)
end

function Event.value(::Type{Tuple{T}}, m::Event.EventMessage) where {T}
    Event.sbe_rewind!(m)
    type = Base.eltype(Event.format(m))
    T <: type || throw(ArgumentError("unexpected type, expected $T got $type"))
    (Event.key_as_string(m), Event.value(T, m))
end

@inline function key!(m::Event.EventMessage, key)
    buf = Event.key(m)
    fill!(buf, 0)
    copyto!(buf, key)
end

function Event.value!(m::Event.EventMessage, key::AbstractString, ::Nothing)
    Event.sbe_rewind!(m)
    Event.format!(m, event_eltype(Nothing))
    key!(m, key)
    Event.skip_value!(m)
    nothing
end

function Event.value!(m::Event.EventMessage, key::AbstractString, value::T) where {T<:Real}
    Event.sbe_rewind!(m)
    Event.format!(m, event_eltype(T))
    key!(m, key)
    Event.value!(T, m, 1)[] = value
end

function Event.value!(m::Event.EventMessage, key::AbstractString, value::T) where {T<:AbstractString}
    Event.sbe_rewind!(m)
    Event.format!(m, event_eltype(T))
    key!(m, key)
    dest = Event.value!(m, length(value))
    copyto!(dest, value)
end

function Event.value!(m::Event.EventMessage, key::AbstractString, value::T) where {T<:AbstractVector{UInt8}}
    Event.sbe_rewind!(m)
    Event.format!(m, event_eltype(T))
    key!(m, key)
    dest = Event.value!(m, length(value))
    copyto!(dest, value)
end

Event.value!(T::Type{<:Real}, m::Event.EventMessage, len::Int) = reinterpret(T, Event.value!(m, sizeof(T) * len))
Event.value!(m::Event.EventMessage, key_value::Tuple{AbstractString,T}) where {T} = Event.value!(m, key_value...)
