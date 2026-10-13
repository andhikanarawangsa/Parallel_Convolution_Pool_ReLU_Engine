// File : maxpool2x2.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Streaming 2x2 max pooling with stride 2 for NUM_CH channels.
// Input arrives in raster order (one value per channel per valid
// cycle). One shared (col,row) counter controls all channels.
// Per channel, the horizontal pair maximum is computed on odd
// columns, stored in a row buffer (IN_W/2 entries) on even rows,
// and combined with the stored value on odd rows, which emits
// one pooled value per 2x2 block.
// Values are signed. The counters count valid inputs only, so
// gaps in the input stream are allowed.
//
// Hierarchy:
// maxpool2x2 : Streaming pooling unit (shared counters,
// per-channel prev register + row buffer)
//
// Latency : 1 clock cycle (registered output).

`timescale 1ns/1ps

module maxpool2x2 #(
    parameter DATA_W = 21,
    parameter IN_W   = 14,
    parameter IN_H   = 14,
    parameter NUM_CH = 3
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        in_valid,
    input  wire [NUM_CH*DATA_W-1:0]    in_data,
    output reg                         out_valid,
    output wire [NUM_CH*DATA_W-1:0]    out_data
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

    localparam CW     = clog2(IN_W);
    localparam RW     = clog2(IN_H);
    localparam HALF_W = IN_W / 2;

    reg [CW-1:0] col;
    reg [RW-1:0] row;

    wire emit = in_valid & col[0] & row[0];

    always @(posedge clk) begin
        if (!rst_n) begin
            col       <= {CW{1'b0}};
            row       <= {RW{1'b0}};
            out_valid <= 1'b0;
        end else begin
            out_valid <= emit;
            if (in_valid) begin
                if (col == IN_W-1) begin
                    col <= {CW{1'b0}};
                    row <= (row == IN_H-1) ? {RW{1'b0}} : row + 1'b1;
                end else begin
                    col <= col + 1'b1;
                end
            end
        end
    end

    genvar g;
    generate
        for (g = 0; g < NUM_CH; g = g + 1) begin : g_ch
            reg  signed [DATA_W-1:0] prev;
            reg  signed [DATA_W-1:0] rowmax [0:HALF_W-1];
            reg  signed [DATA_W-1:0] pool_q;

            wire signed [DATA_W-1:0] x     = in_data[g*DATA_W +: DATA_W];
            wire signed [DATA_W-1:0] hmax  = (prev > x) ? prev : x;
            wire signed [DATA_W-1:0] above = rowmax[col[CW-1:1]];
            wire signed [DATA_W-1:0] vmax  = (above > hmax) ? above : hmax;

            always @(posedge clk) begin
                if (in_valid) begin
                    if (!col[0])
                        prev <= x;
                    else if (!row[0])
                        rowmax[col[CW-1:1]] <= hmax;
                end
                if (emit)
                    pool_q <= vmax;
            end

            assign out_data[g*DATA_W +: DATA_W] = pool_q;
        end
    endgenerate

endmodule
