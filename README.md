# SpidersMessageCodec

```julia
julia> buf = zeros(UInt8, 10000);
julia> data = rand(Int16, 5, 8);

# Instantiate encoder
julia> enc = Tensor.TensorMessageEncoder(buf, Tensor.MessageHeader(buf));
# Non-allocating encoder uses external position pointer
julia> position_ptr = Ref{Int64}()
julia> enc = Tensor.TensorMessageEncoder(buf, position_ptr, Tensor.MessageHeader(buf));
# Instantiate decoder
julia> dec = Tensor.TensorMessageDecoder(buf, Tensor.MessageHeader(buf));

# Tensor with offset (10, 0)
julia> enc(data; offset=(10, 0))
5×8 reshape(reinterpret(Int16, view(::Vector{UInt8}, 105:184)), 5, 8) with eltype Int16:
  -3114   12073  -27310   13451   23514    9311  -25151  10918
  27138  -27909  -17270   21362  -27810  -17819   24535    228
  26611  -10223  -31011    6737   26844  -29897   -4325  15074
 -22652   31731  -17446     318   10359   31078  -10769  29183
 -10071   -8573   -9952  -12523  -13172   17110  -28278   8771

julia> enc
TensorMessage view over a type Vector{UInt8}
SbeBlockLength: 68
SbeTemplateId:  1
SbeSchemaId:    1
SbeSchemaVersion: 0
header: SpidersMessageHeader view over a type Vector{UInt8}
channelRcvTimestampNs: 0
channelSndTimestampNs: 0
TimestampNs: 0
correlationId: 0
tag: ""
format: INT16
order: COLUMN
dims: (5, 8)
offset: (10, 0)
values: 80 bytes of raw data

# Adjoint
julia> enc(data'; offset=(10, 0))
8×5 reshape(reinterpret(Int16, view(::Vector{UInt8}, 105:184)), 8, 5) with eltype Int16:
  -3114   27138   26611  -22652  -10071
  12073  -27909  -10223   31731   -8573
 -27310  -17270  -31011  -17446   -9952
  13451   21362    6737     318  -12523
  23514  -27810   26844   10359  -13172
   9311  -17819  -29897   31078   17110
 -25151   24535   -4325  -10769  -28278
  10918     228   15074   29183    8771

julia> enc
TensorMessage view over a type Vector{UInt8}
SbeBlockLength: 68
SbeTemplateId:  1
SbeSchemaId:    1
SbeSchemaVersion: 0
header: SpidersMessageHeader view over a type Vector{UInt8}
channelRcvTimestampNs: 0
channelSndTimestampNs: 0
TimestampNs: 0
correlationId: 0
tag: ""
format: INT16
order: ROW
dims: (8, 5)
offset: (10, 0)
values: 80 bytes of raw data

# Type-unstable view into buffer 
julia> dec()
5×8 reshape(reinterpret(Int16, view(::Vector{UInt8}, 105:184)), 5, 8) with eltype Int16:
  -3114   12073  -27310   13451   23514    9311  -25151  10918
  27138  -27909  -17270   21362  -27810  -17819   24535    228
  26611  -10223  -31011    6737   26844  -29897   -4325  15074
 -22652   31731  -17446     318   10359   31078  -10769  29183
 -10071   -8573   -9952  -12523  -13172   17110  -28278   8771

# Type-stable view into buffer
julia> dec(Matrix{Int16})
5×8 reshape(reinterpret(Int16, view(::Vector{UInt8}, 105:184)), 5, 8) with eltype Int16:
  -3114   12073  -27310   13451   23514    9311  -25151  10918
  27138  -27909  -17270   21362  -27810  -17819   24535    228
  26611  -10223  -31011    6737   26844  -29897   -4325  15074
 -22652   31731  -17446     318   10359   31078  -10769  29183
 -10071   -8573   -9952  -12523  -13172   17110  -28278   8771

```

# Save to file (could be sent over network, mmap, Aeron, etc.) and recover:

```julia
julia> write("tmp.dat", buf)

julia> buf2 = read("tmp.dat")
julia> dec = Tensor.TensorMessageDecoder(buf2, Tensor.MessageHeader(buf))
julia> Tensor.skip_dims!(dec)
# Type-stable offset
julia> offset2 = Tensor.offset(NTuple{2}, dec)
(5, 8)

# Type-stable img
julia> img2 = dec(Matrix{Int16})
5×8 reshape(reinterpret(Int16, view(::Vector{UInt8}, 105:184)), 5, 8) with eltype Int16:
  -3114   12073  -27310   13451   23514    9311  -25151  10918
  27138  -27909  -17270   21362  -27810  -17819   24535    228
  26611  -10223  -31011    6737   26844  -29897   -4325  15074
 -22652   31731  -17446     318   10359   31078  -10769  29183
 -10071   -8573   -9952  -12523  -13172   17110  -28278   8771

```