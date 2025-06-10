using Test
using SpidersMessageCodecs

@testset "Message Headers and Metadata Tests" begin
    @testset "SpidersMessageHeader" begin
        buf = zeros(UInt8, 1000)
        
        header = SpidersMessageHeader(buf, 0, 0)
        
        # Test basic creation
        @test header isa SpidersMessageHeader
        @test SpidersMessageCodecs.sbe_buffer(header) === buf
        @test SpidersMessageCodecs.sbe_offset(header) == 0
        
        # Test timestamp fields
        SpidersMessageCodecs.channelRcvTimestampNs!(header, Int64(1234567890))
        SpidersMessageCodecs.channelSndTimestampNs!(header, Int64(9876543210))
        SpidersMessageCodecs.timestampNs!(header, Int64(555666777))
        SpidersMessageCodecs.correlationId!(header, Int64(42))
        
        timestamp_rcv = SpidersMessageCodecs.channelRcvTimestampNs(header)
        timestamp_snd = SpidersMessageCodecs.channelSndTimestampNs(header)
        timestamp = SpidersMessageCodecs.timestampNs(header)
        correlation = SpidersMessageCodecs.correlationId(header)
        
        @test timestamp_rcv == Int64(1234567890)
        @test timestamp_snd == Int64(9876543210)
        @test timestamp == Int64(555666777)
        @test correlation == Int64(42)
        
        # Test tag field
        test_tag = "MyTestTag"
        SpidersMessageCodecs.tag!(header, test_tag)
        retrieved_tag = SpidersMessageCodecs.tag(header, String)
        @test String(retrieved_tag) == test_tag
    end
    
    @testset "MessageHeader" begin
        buf = zeros(UInt8, 100)
        
        header = MessageHeader(buf, 0, 0)
        
        # Test basic creation
        @test header isa MessageHeader
        @test SpidersMessageCodecs.sbe_buffer(header) === buf
        @test SpidersMessageCodecs.sbe_offset(header) == 0
        
        # Test header fields
        SpidersMessageCodecs.blockLength!(header, UInt16(1000))
        SpidersMessageCodecs.templateId!(header, UInt16(42))
        SpidersMessageCodecs.schemaId!(header, UInt16(1))
        SpidersMessageCodecs.version!(header, UInt16(0))
        
        block_length = SpidersMessageCodecs.blockLength(header)
        template_id = SpidersMessageCodecs.templateId(header)
        schema_id = SpidersMessageCodecs.schemaId(header)
        version = SpidersMessageCodecs.version(header)
        
        @test block_length == UInt16(1000)
        @test template_id == UInt16(42)
        @test schema_id == UInt16(1)
        @test version == UInt16(0)
    end
    
    @testset "Variable Length Encodings" begin
        @testset "VarStringEncoding" begin
            buf = zeros(UInt8, 1000)
            
            encoder = VarStringEncoding(buf, 0, 0)
            decoder = VarStringEncoding(buf, 0, 0)
            
            # Test string length setting
            test_string = "Hello, Variable String!"
            string_length = UInt32(length(test_string))
            
            SpidersMessageCodecs.length!(encoder, string_length)
            retrieved_length = SpidersMessageCodecs.length(decoder)
            
            @test retrieved_length == string_length
        end
        
        @testset "VarDataEncoding" begin
            buf = zeros(UInt8, 1000)
            
            encoder = VarDataEncoding(buf, 0, 0)
            decoder = VarDataEncoding(buf, 0, 0)
            
            # Test data length setting
            data_length = UInt32(1024)
            
            SpidersMessageCodecs.length!(encoder, data_length)
            retrieved_length = SpidersMessageCodecs.length(decoder)
            
            @test retrieved_length == data_length
        end
        
        @testset "GroupSizeEncoding" begin
            buf = zeros(UInt8, 100)
            
            encoder = GroupSizeEncoding(buf, 0, 0)
            decoder = GroupSizeEncoding(buf, 0, 0)
            
            # Test group size fields
            SpidersMessageCodecs.blockLength!(encoder, UInt16(256))
            SpidersMessageCodecs.numInGroup!(encoder, UInt16(10))
            
            block_length = SpidersMessageCodecs.blockLength(decoder)
            num_in_group = SpidersMessageCodecs.numInGroup(decoder)
            
            @test block_length == UInt16(256)
            @test num_in_group == UInt16(10)
        end
    end
    
    @testset "Format and Order Enums" begin
        @testset "Format Enum Values" begin
            # Test all format enum values - compare to their actual Int8 values
            @test Int8(SpidersMessageCodecs.Format.NOTHING) == Int8(0)
            @test Int8(SpidersMessageCodecs.Format.UINT8) == Int8(1)
            @test Int8(SpidersMessageCodecs.Format.INT8) == Int8(2)
            @test Int8(SpidersMessageCodecs.Format.UINT16) == Int8(3)
            @test Int8(SpidersMessageCodecs.Format.INT16) == Int8(4)
            @test Int8(SpidersMessageCodecs.Format.UINT32) == Int8(5)
            @test Int8(SpidersMessageCodecs.Format.INT32) == Int8(6)
            @test Int8(SpidersMessageCodecs.Format.UINT64) == Int8(7)
            @test Int8(SpidersMessageCodecs.Format.INT64) == Int8(8)
            @test Int8(SpidersMessageCodecs.Format.FLOAT32) == Int8(9)
            @test Int8(SpidersMessageCodecs.Format.FLOAT64) == Int8(10)
            @test Int8(SpidersMessageCodecs.Format.BOOLEAN) == Int8(11)
            @test Int8(SpidersMessageCodecs.Format.STRING) == Int8(12)
            @test Int8(SpidersMessageCodecs.Format.BYTES) == Int8(13)
            @test Int8(SpidersMessageCodecs.Format.BIT) == Int8(14)
            @test Int8(SpidersMessageCodecs.Format.SBE) == Int8(15)
        end
        
        @testset "MajorOrder Enum Values" begin
            @test Int8(SpidersMessageCodecs.MajorOrder.ROW) == Int8(0)
            @test Int8(SpidersMessageCodecs.MajorOrder.COLUMN) == Int8(1)
        end
        
        @testset "Indexing Enum Values" begin
            @test Int8(SpidersMessageCodecs.Indexing.ZERO) == Int8(0)
            @test Int8(SpidersMessageCodecs.Indexing.ONE) == Int8(1)
        end
    end
end
