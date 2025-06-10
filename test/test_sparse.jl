using Test
using SpidersMessageCodecs

@testset "Sparse Matrix Tests" begin
    @testset "SparseMatrixCSXMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = SparseMatrixCSXMessageEncoder(buf; position_ptr)
        dec = SparseMatrixCSXMessageDecoder(buf; position_ptr)
        
        # Test basic encoding/decoding
        position_ptr[] = 0
        
        # Verify we can create the encoder/decoder
        @test enc isa SparseMatrixCSXMessageEncoder
        @test dec isa SparseMatrixCSXMessageDecoder
        
        # Test format setting
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.format!(enc, SpidersMessageCodecs.Format.INT32)
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        format = SpidersMessageCodecs.format(dec)
        @test format == SpidersMessageCodecs.Format.INT32
    end
    
    @testset "SparseVectorMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = SparseVectorMessageEncoder(buf; position_ptr)
        dec = SparseVectorMessageDecoder(buf; position_ptr)
        
        # Test basic encoding/decoding
        position_ptr[] = 0
        
        # Verify we can create the encoder/decoder
        @test enc isa SparseVectorMessageEncoder
        @test dec isa SparseVectorMessageDecoder
        
        # Test format and indices format setting
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.format!(enc, SpidersMessageCodecs.Format.FLOAT32)
        SpidersMessageCodecs.indiciesFormat!(enc, SpidersMessageCodecs.Format.INT32)
        SpidersMessageCodecs.indexing!(enc, SpidersMessageCodecs.Indexing.ZERO)
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        format = SpidersMessageCodecs.format(dec)
        indices_format = SpidersMessageCodecs.indiciesFormat(dec)
        indexing = SpidersMessageCodecs.indexing(dec)
        
        @test format == SpidersMessageCodecs.Format.FLOAT32
        @test indices_format == SpidersMessageCodecs.Format.INT32
        @test indexing == SpidersMessageCodecs.Indexing.ZERO
    end
    
    @testset "DiagonalMatrixMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = DiagonalMatrixMessageEncoder(buf; position_ptr)
        dec = DiagonalMatrixMessageDecoder(buf; position_ptr)
        
        # Test basic encoding/decoding
        position_ptr[] = 0
        
        # Verify we can create the encoder/decoder
        @test enc isa DiagonalMatrixMessageEncoder
        @test dec isa DiagonalMatrixMessageDecoder
        
        # Test format setting
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.format!(enc, SpidersMessageCodecs.Format.FLOAT64)
        SpidersMessageCodecs.indiciesFormat!(enc, SpidersMessageCodecs.Format.INT64)
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        format = SpidersMessageCodecs.format(dec)
        indices_format = SpidersMessageCodecs.indiciesFormat(dec)
        
        @test format == SpidersMessageCodecs.Format.FLOAT64
        @test indices_format == SpidersMessageCodecs.Format.INT64
    end
end
