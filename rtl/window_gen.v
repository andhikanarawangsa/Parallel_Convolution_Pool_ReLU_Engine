// File : window_gen.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Sliding-window generator. Receives the image as a raster-scan
// pixel stream (1 pixel per valid cycle) and produces a KxK
// window every time a full window is available (stride 1,
// valid padding). K-1 line buffers hold the previous rows so
// every pixel is read from the input only once. win_data is
// packed row-major: element (r,c) = win_data[(r*K+c)*DATA_W +: DATA_W]
// with r = 0 the oldest row and c = 0 the leftmost column.
//
// Hierarchy:
// window_gen : Sliding-window generator
// line_buffer : (K-1 instances) one image row delay each
//
// Latency : 1 clock cycle (window registers) after the pixel that
// completes the window is accepted.

`timescale 1ns/1ps

module window_gen #(
    parameter DATA_W = 8,
    parameter IMG_W  = 16,
    parameter IMG_H  = 16,
    parameter K      = 3
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      pix_valid,
    input  wire [DATA_W-1:0]         pix_data,
    output reg                       win_valid,
    output wire [K*K*DATA_W-1:0]     win_data
);

    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            clog2 = 0;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v = v >> 1;
            end
            if (clog2 < 1) clog2 = 1;
        end
    endfunction

    localparam CW = clog2(IMG_W);
    localparam RW = clog2(IMG_H);

    // tap[K-1] = current row, tap[K-2] = one row earlier, ... tap[0] = K-1 rows earlier
    wire [DATA_W-1:0] tap [0:K-1];
    assign tap[K-1] = pix_data;

    genvar m;
    generate
        for (m = 0; m < K-1; m = m + 1) begin : g_lb
            line_buffer #(.DATA_W(DATA_W), .DEPTH(IMG_W)) u_lb (
                .clk  (clk),
                .en   (pix_valid),
                .din  (tap[K-1-m]),
                .dout (tap[K-2-m])
            );
        end
    endgenerate

    // KxK window shift registers (shift left on every accepted pixel)
    reg [DATA_W-1:0] win [0:K*K-1];
    integer r, c;

    always @(posedge clk) begin
        if (pix_valid) begin
            for (r = 0; r < K; r = r + 1) begin
                for (c = 0; c < K-1; c = c + 1)
                    win[r*K+c] <= win[r*K+c+1];
                win[r*K+K-1] <= tap[r];
            end
        end
    end

    genvar w;
    generate
        for (w = 0; w < K*K; w = w + 1) begin : g_out
            assign win_data[w*DATA_W +: DATA_W] = win[w];
        end
    endgenerate

    // Position counters of the incoming pixel -> mask for a full window
    reg [CW-1:0] col;
    reg [RW-1:0] row;

    always @(posedge clk) begin
        if (!rst_n) begin
            col       <= {CW{1'b0}};
            row       <= {RW{1'b0}};
            win_valid <= 1'b0;
        end else begin
            win_valid <= 1'b0;
            if (pix_valid) begin
                win_valid <= (row >= K-1) && (col >= K-1);
                if (col == IMG_W-1) begin
                    col <= {CW{1'b0}};
                    row <= (row == IMG_H-1) ? {RW{1'b0}} : row + 1'b1;
                end else begin
                    col <= col + 1'b1;
                end
            end
        end
    end

endmodule
