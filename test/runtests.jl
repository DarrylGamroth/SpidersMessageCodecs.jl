using Test

# Main comprehensive test suite
@testset "SpidersMessageCodecs.jl Comprehensive Tests" begin
    
    # Include the main functionality tests
    include("test_core.jl")
    
    # Include specialized component tests
    include("test_sparse.jl")
    include("test_streaming.jl") 
    include("test_headers.jl")
    include("test_performance.jl")
    
end
