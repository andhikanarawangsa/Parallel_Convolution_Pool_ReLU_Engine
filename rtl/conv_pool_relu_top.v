// File : conv_pool_relu_top.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Streaming parallel pipeline with X input size of 16x16, K kernel
// size of 3x3 with 3 kernels, conv output of 14x14x3, maxpool 2x2
// stride 2 output of 7x7x3, then ReLU and saturation to 8-bit
// unsigned. Stride 1, no padding. One pixel is accepted per
// clock; the 3 kernels run in parallel on a shared sliding
// window with 9 parallel multipliers each (27 in total), and
// window / convolution / pooling / ReLU stages overlap in a
// pipeline. The 3 output channels leave in the same cycle.
//
// Hierarchy:
// conv_pool_relu_top : Top-level connecting all stages
// window_gen : Line buffers + 3x3 sliding window
// conv_array : 3 parallel conv_pe (9 multipliers each)
// maxpool2x2 : Streaming 2x2 / stride 2 max pooling
// relu_sat : ReLU + saturation to 8-bit unsigned
// frame_ctrl : Busy / done frame status
//
// Computation latency: 261 clock cycles without stall (256 cycles to stream
// the input + 5 pipeline cycles).

`timescale 1ns/1ps

module conv_pool_relu_top #(
    parameter DATA_W = 8,
    parameter COEF_W = 8,
    parameter IMG_W  = 16,
    parameter IMG_H  = 16,
    parameter K      = 3,
    parameter NUM_K  = 3,
    parameter OUT_W  = 8,
    parameter OUT_SHIFT = 0,   // optional right shift before 8-bit saturation
    // Accumulator width = DATA_W + COEF_W + 1 + ceil(log2(K*K)); default 21 for K=3
    parameter ACC_W  = DATA_W + COEF_W + 1 +
                       ((K*K<=2) ? 1 : (K*K<=4) ? 2 : (K*K<=8) ? 3 :
                        (K*K<=16) ? 4 : (K*K<=32) ? 5 : 6)
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // Input pixel stream (raster order) and kernel coefficients
    input  wire                              pix_valid,
    input  wire [DATA_W-1:0]                 pix_data,
    input  wire [NUM_K*K*K*COEF_W-1:0]       kernel_flat,

    // Final output: one beat = NUM_K channel values of one 2x2 block
    output wire                              out_valid,
    output wire [NUM_K*OUT_W-1:0]            out_data,

    // Frame status
    output wire                              busy,
    output wire                              done,

    // Monitoring ports (intermediate results, for simulation / debug)
    output wire                              conv_valid,
    output wire [NUM_K*ACC_W-1:0] conv_data,
    output wire                              pool_valid,
    output wire [NUM_K*ACC_W-1:0] pool_data,
    output wire                              relu_valid,
    output wire [NUM_K*ACC_W-1:0] relu_data
);

    localparam CONV_W   = IMG_W - K + 1;
    localparam CONV_H   = IMG_H - K + 1;
    localparam POOL_W   = CONV_W / 2;
    localparam POOL_H   = CONV_H / 2;

    // Stage 1: sliding window
    wire                    win_valid;
    wire [K*K*DATA_W-1:0]   win_data;

    window_gen #(
        .DATA_W (DATA_W),
        .IMG_W  (IMG_W),
        .IMG_H  (IMG_H),
        .K      (K)
    ) u_window (
        .clk       (clk),
        .rst_n     (rst_n),
        .pix_valid (pix_valid),
        .pix_data  (pix_data),
        .win_valid (win_valid),
        .win_data  (win_data)
    );

    // Stage 2: parallel convolution
    conv_array #(
        .DATA_W (DATA_W),
        .COEF_W (COEF_W),
        .K      (K),
        .NUM_K  (NUM_K),
        .ACC_W  (ACC_W)
    ) u_conv (
        .clk         (clk),
        .rst_n       (rst_n),
        .win_valid   (win_valid),
        .win_data    (win_data),
        .kernel_flat (kernel_flat),
        .conv_valid  (conv_valid),
        .conv_data   (conv_data)
    );

    // Stage 3: 2x2 max pooling
    maxpool2x2 #(
        .DATA_W (ACC_W),
        .IN_W   (CONV_W),
        .IN_H   (CONV_H),
        .NUM_CH (NUM_K)
    ) u_pool (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (conv_valid),
        .in_data   (conv_data),
        .out_valid (pool_valid),
        .out_data  (pool_data)
    );

    // Stage 4: ReLU + saturation
    relu_sat #(
        .IN_W     (ACC_W),
        .OUT_W    (OUT_W),
        .NUM_CH   (NUM_K),
        .SATURATE (1),
        .OUT_SHIFT(OUT_SHIFT)
    ) u_relu (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (pool_valid),
        .in_data   (pool_data),
        .out_valid (relu_valid),
        .relu_data (relu_data),
        .out_data  (out_data)
    );

    assign out_valid = relu_valid;

    // Frame status
    frame_ctrl #(
        .OUT_BEATS (POOL_W * POOL_H)
    ) u_ctrl (
        .clk       (clk),
        .rst_n     (rst_n),
        .pix_valid (pix_valid),
        .out_valid (relu_valid),
        .busy      (busy),
        .done      (done)
    );

endmodule
