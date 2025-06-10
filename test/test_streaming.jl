using Test
using SpidersMessageCodecs

@testset "Streaming and Chunking Tests" begin
    @testset "TensorStreamHeaderMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = TensorStreamHeaderMessageEncoder(buf; position_ptr)
        dec = TensorStreamHeaderMessageDecoder(buf; position_ptr)
        
        # Test basic creation
        @test enc isa TensorStreamHeaderMessageEncoder
        @test dec isa TensorStreamHeaderMessageDecoder
        
        # Test header fields
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.format!(enc, SpidersMessageCodecs.Format.FLOAT32)
        SpidersMessageCodecs.majorOrder!(enc, SpidersMessageCodecs.MajorOrder.COLUMN)
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        format = SpidersMessageCodecs.format(dec)
        major_order = SpidersMessageCodecs.majorOrder(dec)
        
        @test format == SpidersMessageCodecs.Format.FLOAT32
        @test major_order == SpidersMessageCodecs.MajorOrder.COLUMN
    end
    
    @testset "TensorStreamDataMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = TensorStreamDataMessageEncoder(buf; position_ptr)
        dec = TensorStreamDataMessageDecoder(buf; position_ptr)
        
        # Test basic creation
        @test enc isa TensorStreamDataMessageEncoder
        @test dec isa TensorStreamDataMessageDecoder
        
        # Test sequence number setting
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.sequencenumber!(enc, Int64(12345))
        SpidersMessageCodecs.offset!(enc, UInt64(67890))
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        sequence = SpidersMessageCodecs.sequencenumber(dec)
        offset = SpidersMessageCodecs.offset(dec)
        
        @test sequence == Int64(12345)
        @test offset == UInt64(67890)
    end
    
    @testset "ChunkHeaderMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = ChunkHeaderMessageEncoder(buf; position_ptr)
        dec = ChunkHeaderMessageDecoder(buf; position_ptr)
        
        # Test basic creation
        @test enc isa ChunkHeaderMessageEncoder
        @test dec isa ChunkHeaderMessageDecoder
        
        # Test chunk header fields
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.sequencenumber!(enc, Int64(999))
        SpidersMessageCodecs.length!(enc, Int64(1024))
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        sequence = SpidersMessageCodecs.sequencenumber(dec)
        length_val = SpidersMessageCodecs.length(dec)
        
        @test sequence == Int64(999)
        @test length_val == Int64(1024)
    end
    
    @testset "ChunkDataMessage" begin
        buf = zeros(UInt8, 10000)
        position_ptr = Ref{Int64}(0)
        
        enc = ChunkDataMessageEncoder(buf; position_ptr)
        dec = ChunkDataMessageDecoder(buf; position_ptr)
        
        # Test basic creation
        @test enc isa ChunkDataMessageEncoder
        @test dec isa ChunkDataMessageDecoder
        
        # Test chunk data fields
        SpidersMessageCodecs.sbe_rewind!(enc)
        SpidersMessageCodecs.sequencenumber!(enc, Int64(777))
        SpidersMessageCodecs.offset!(enc, Int64(2048))
        
        SpidersMessageCodecs.sbe_rewind!(dec)
        sequence = SpidersMessageCodecs.sequencenumber(dec)
        offset = SpidersMessageCodecs.offset(dec)
        
        @test sequence == Int64(777)
        @test offset == Int64(2048)
    end
end
