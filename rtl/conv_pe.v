// File : conv_pe.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Fully parallel convolution processing element for ONE kernel.
// Stage 1 : K*K multipliers work in parallel (pixel as unsigned,
// coefficient as signed) and the products are registered.
// Stage 2 : K*K products are summed by a combinational adder
// tree and the sum is registered (signed, ACC_W bits).
// Kernel is applied without flipping (cross-correlation, as in CNNs).
//
// Hierarchy:
// conv_pe : Leaf module (no sub-instances)
//
// Latency : 2 clock cycles, throughput 1 result per cycle.

`timescale 1ns/1ps

module conv_pe #(
    parameter DATA_W = 8,
    parameter COEF_W = 8,
    parameter K      = 3,
    parameter ACC_W  = 21
) (
    input  wire                         clk,
    input  wire [K*K*DATA_W-1:0]        win_data,
    input  wire [K*K*COEF_W-1:0]        coef,
    output reg  signed [ACC_W-1:0]      conv_out
);

    localparam N      = K*K;
    localparam PROD_W = DATA_W + COEF_W + 1;

    // Stage 1: parallel multipliers
    reg signed [PROD_W-1:0] prod [0:N-1];
    integer i_p;

    always @(posedge clk) begin
        for (i_p = 0; i_p < N; i_p = i_p + 1)
            prod[i_p] <= $signed({1'b0, win_data[i_p*DATA_W +: DATA_W]}) *
                         $signed(coef[i_p*COEF_W +: COEF_W]);
    end

    // Stage 2: adder tree
    reg signed [ACC_W-1:0] sum_c;
    integer i_s;

    always @* begin
        sum_c = {ACC_W{1'b0}};
        for (i_s = 0; i_s < N; i_s = i_s + 1)
            sum_c = sum_c + prod[i_s];
    end

    always @(posedge clk) begin
        conv_out <= sum_c;
    end

endmodule
