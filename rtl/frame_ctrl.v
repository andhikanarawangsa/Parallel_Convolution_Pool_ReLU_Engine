// File : frame_ctrl.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Frame-level status controller. Raises busy on the first
// accepted pixel of a frame and, after OUT_BEATS output beats
// have been observed, drops busy and gives a one-cycle done
// pulse. The datapath itself is data-driven (valid flags), so
// this block only reports frame status. A new frame should be
// started after done.
//
// Hierarchy:
// frame_ctrl : Leaf module (no sub-instances)
//
// Latency : done is asserted 1 clock cycle after the last output beat.

`timescale 1ns/1ps

module frame_ctrl #(
    parameter OUT_BEATS = 49
) (
    input  wire clk,
    input  wire rst_n,
    input  wire pix_valid,
    input  wire out_valid,
    output reg  busy,
    output reg  done
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

    localparam CNT_W = clog2(OUT_BEATS + 1);

    reg [CNT_W-1:0] out_cnt;
    wire last_beat = out_valid && (out_cnt == OUT_BEATS-1);

    always @(posedge clk) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            out_cnt <= {CNT_W{1'b0}};
        end else begin
            done <= 1'b0;
            if (pix_valid)
                busy <= 1'b1;
            if (out_valid) begin
                if (last_beat) out_cnt <= {CNT_W{1'b0}};
                else           out_cnt <= out_cnt + 1'b1;
            end
            if (last_beat) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end

endmodule
