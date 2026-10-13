// File : line_buffer.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Parameterizable shift-register line buffer. Stores one image
// row of DEPTH pixels. Data advances by one position only
// when en is high, so the buffer tolerates input stalls.
// dout is din delayed by DEPTH accepted samples.
//
// Hierarchy:
// line_buffer : Leaf module (no sub-instances)
//
// Latency : DEPTH accepted samples (en pulses).

`timescale 1ns/1ps

module line_buffer #(
    parameter DATA_W = 8,
    parameter DEPTH  = 16
) (
    input  wire                clk,
    input  wire                en,
    input  wire [DATA_W-1:0]   din,
    output wire [DATA_W-1:0]   dout
);

    reg [DATA_W-1:0] sr [0:DEPTH-1];
    integer i;

    always @(posedge clk) begin
        if (en) begin
            sr[0] <= din;
            for (i = 1; i < DEPTH; i = i + 1)
                sr[i] <= sr[i-1];
        end
    end

    assign dout = sr[DEPTH-1];

endmodule
