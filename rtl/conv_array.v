// File : conv_array.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Array of NUM_K convolution PEs that share the same window.
// All kernels are computed simultaneously (spatial parallelism),
// so the 3 output channels appear in the same clock cycle.
// Also carries the valid flag through the PE pipeline.
// Packing: kernel_flat[(k*K*K + i)*COEF_W +: COEF_W] is element i
// (row-major) of kernel k; conv_data[k*ACC_W +: ACC_W] is the
// signed result of kernel k.
//
// Hierarchy:
// conv_array : Parallel kernel array + valid pipeline
// conv_pe : (NUM_K instances) one 3x3 MAC engine per kernel
//
// Latency : 2 clock cycles (same as conv_pe).

`timescale 1ns/1ps

module conv_array #(
    parameter DATA_W = 8,
    parameter COEF_W = 8,
    parameter K      = 3,
    parameter NUM_K  = 3,
    parameter ACC_W  = 21
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              win_valid,
    input  wire [K*K*DATA_W-1:0]             win_data,
    input  wire [NUM_K*K*K*COEF_W-1:0]       kernel_flat,
    output wire                              conv_valid,
    output wire [NUM_K*ACC_W-1:0]            conv_data
);

    genvar k;
    generate
        for (k = 0; k < NUM_K; k = k + 1) begin : g_pe
            wire signed [ACC_W-1:0] pe_out;

            conv_pe #(
                .DATA_W (DATA_W),
                .COEF_W (COEF_W),
                .K      (K),
                .ACC_W  (ACC_W)
            ) u_pe (
                .clk      (clk),
                .win_data (win_data),
                .coef     (kernel_flat[k*K*K*COEF_W +: K*K*COEF_W]),
                .conv_out (pe_out)
            );

            assign conv_data[k*ACC_W +: ACC_W] = pe_out;
        end
    endgenerate

    // Valid flag follows the 2-stage PE pipeline
    reg [1:0] vpipe;

    always @(posedge clk) begin
        if (!rst_n) vpipe <= 2'b00;
        else        vpipe <= {vpipe[0], win_valid};
    end

    assign conv_valid = vpipe[1];

endmodule
