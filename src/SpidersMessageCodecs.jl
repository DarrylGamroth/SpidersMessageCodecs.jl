module SpidersMessageCodecs

using LinearAlgebra
using StringViews
using UnsafeArrays
using ValSplit

include("Sbe.jl")

# Include SBE generated code
include("Tensor.jl")

# Include Event last
include("Event.jl")


export Tensor, Event
end # module SpidersMessageCodecs