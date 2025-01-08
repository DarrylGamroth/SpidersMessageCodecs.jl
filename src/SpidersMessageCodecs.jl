module SpidersMessageCodecs

using UnsafeArrays
using ValSplit

include("spidersheader/Spidersheader.jl")
include("event/Event.jl")
include("tensor/Tensor.jl")

include("sbe.jl")

include("utils.jl")

using .Spidersheader
export Spidersheader

using .Event
export Event

using .Tensor
export Tensor

using .Sbe
export Sbe

include("../ext/EventEx.jl")
using .EventEx
export EventEx

include("../ext/TensorEx.jl")
using .TensorEx
export TensorEx

end # module SpidersMessageCodecs