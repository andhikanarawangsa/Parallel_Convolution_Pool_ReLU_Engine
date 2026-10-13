// File : relu_sat.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Per-element ReLU, f(x) = x if x >= 0 else 0, followed by
// saturation to OUT_W bits unsigned (clamp to 2^OUT_W - 1).
// relu_data keeps the full precision ReLU result (for monitoring),
// out_data is the final OUT_W-bit unsigned output:
// out_data = saturate( relu_data >> OUT_SHIFT )
// OUT_SHIFT (default 0) is an optional power-of-two scaling to fit
// the wide accumulator range into 8 bits. SATURATE = 0 selects
// plain truncation of the low OUT_W bits instead of clamping.
//
// Hierarchy:
// relu_sat : Leaf module (no sub-instances)
//
// Latency : 1 clock cycle (registered output).

`timescale 1ns/1ps

module relu_sat #(
    parameter IN_W     = 21,
    parameter OUT_W    = 8,
    parameter NUM_CH   = 3,
    parameter SATURATE = 1,
    parameter OUT_SHIFT = 0
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        in_valid,
    input  wire [NUM_CH*IN_W-1:0]      in_data,
    output reg                         out_valid,
    output wire [NUM_CH*IN_W-1:0]      relu_data,
    output wire [NUM_CH*OUT_W-1:0]     out_data
);

    always @(posedge clk) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= in_valid;
    end

    genvar g;
    generate
        for (g = 0; g < NUM_CH; g = g + 1) begin : g_ch
            wire [IN_W-1:0]  x    = in_data[g*IN_W +: IN_W];
            wire [IN_W-1:0]  r    = x[IN_W-1] ? {IN_W{1'b0}} : x;
            wire [IN_W-1:0]  rs   = r >> OUT_SHIFT;
            wire             ovf  = |rs[IN_W-1:OUT_W];
            wire [OUT_W-1:0] s    = (SATURATE != 0 && ovf) ? {OUT_W{1'b1}} : rs[OUT_W-1:0];

            reg [IN_W-1:0]  r_q;
            reg [OUT_W-1:0] s_q;

            always @(posedge clk) begin
                if (in_valid) begin
                    r_q <= r;
                    s_q <= s;
                end
            end

            assign relu_data[g*IN_W +: IN_W]   = r_q;
            assign out_data[g*OUT_W +: OUT_W]  = s_q;
        end
    endgenerate

endmodule
