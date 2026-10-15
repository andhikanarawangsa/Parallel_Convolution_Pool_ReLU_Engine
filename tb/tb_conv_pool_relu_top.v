// File : tb_conv_pool_relu_top.v
// Author : Andhika Narawangsa Susilo - 13222036
// Project : Streaming Convolution + Maxpooling + ReLU Accelerator
// Description : Self-checking testbench for conv_pool_relu_top.
// Test 1 : main test vector from the assignment (image.mem,
// kernel.mem), back-to-back pixels. Prints input,
// convolution (3 channels), pooling, ReLU and final output.
// Test 2 : corner case (all pixels 255, extreme kernels) for
// accumulator range and saturation.
// Test 3 : random frames with random kernels, some with random
// input stalls (pix_valid gaps).
// Every stage (conv, pool, relu, final) is compared with a
// behavioural reference model written with plain loops.
//
// Hierarchy:
// tb_conv_pool_relu_top : Testbench
// conv_pool_relu_top : Device under test
//
// Run from the repository root (vector paths are relative to it).

`timescale 1ns/1ps

module tb_conv_pool_relu_top;

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
        end
    endfunction

    localparam DATA_W = 8;
    localparam COEF_W = 8;
    localparam IMG_W  = 16;
    localparam IMG_H  = 16;
    localparam K      = 3;
    localparam NUM_K  = 3;
    localparam OUT_W  = 8;
    parameter  OUT_SHIFT = 0;   // override: iverilog -Ptb_conv_pool_relu_top.OUT_SHIFT=8
    localparam ACC_W  = DATA_W + COEF_W + 1 + clog2(K*K);
    localparam CONV_W = IMG_W - K + 1;
    localparam CONV_H = IMG_H - K + 1;
    localparam POOL_W = CONV_W / 2;
    localparam POOL_H = CONV_H / 2;
    localparam N_IMG  = IMG_W * IMG_H;
    localparam N_CONV = NUM_K * CONV_W * CONV_H;
    localparam N_POOL = NUM_K * POOL_W * POOL_H;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    reg                                pix_valid = 1'b0;
    reg  [DATA_W-1:0]                  pix_data  = {DATA_W{1'b0}};
    reg  [NUM_K*K*K*COEF_W-1:0]        kernel_flat = {(NUM_K*K*K*COEF_W){1'b0}};

    wire                               out_valid;
    wire [NUM_K*OUT_W-1:0]             out_data;
    wire                               busy, done;
    wire                               conv_valid, pool_valid, relu_valid;
    wire [NUM_K*ACC_W-1:0]             conv_data, pool_data, relu_data;

    conv_pool_relu_top #(
        .DATA_W (DATA_W), .COEF_W (COEF_W),
        .IMG_W  (IMG_W),  .IMG_H  (IMG_H),
        .K      (K),      .NUM_K  (NUM_K),
        .OUT_W  (OUT_W), .OUT_SHIFT (OUT_SHIFT)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .pix_valid(pix_valid), .pix_data(pix_data), .kernel_flat(kernel_flat),
        .out_valid(out_valid), .out_data(out_data),
        .busy(busy), .done(done),
        .conv_valid(conv_valid), .conv_data(conv_data),
        .pool_valid(pool_valid), .pool_data(pool_data),
        .relu_valid(relu_valid), .relu_data(relu_data)
    );

    // ------------------------------------------------------------------
    // Test data and reference results
    // ------------------------------------------------------------------
    reg [DATA_W-1:0] img  [0:N_IMG-1];
    reg [COEF_W-1:0] kmem [0:NUM_K*K*K-1];

    integer exp_conv [0:N_CONV-1];
    integer exp_pool [0:N_POOL-1];
    integer exp_relu [0:N_POOL-1];
    integer exp_out  [0:N_POOL-1];

    integer got_conv [0:N_CONV-1];
    integer got_pool [0:N_POOL-1];
    integer got_relu [0:N_POOL-1];
    integer got_out  [0:N_POOL-1];

    integer total_errors;
    integer frame_errors;

    // Reference model (plain loops, independent of the RTL structure)
    task compute_expected;
        integer k, r, c, i, j, acc, pix, kv, m, a, b, d, e;
        begin
            for (k = 0; k < NUM_K; k = k + 1)
                for (r = 0; r < CONV_H; r = r + 1)
                    for (c = 0; c < CONV_W; c = c + 1) begin
                        acc = 0;
                        for (i = 0; i < K; i = i + 1)
                            for (j = 0; j < K; j = j + 1) begin
                                pix = img[(r+i)*IMG_W + c + j];
                                kv  = kmem[k*K*K + i*K + j];
                                if (kv >= (1 << (COEF_W-1))) kv = kv - (1 << COEF_W);
                                acc = acc + pix * kv;
                            end
                        exp_conv[k*CONV_W*CONV_H + r*CONV_W + c] = acc;
                    end
            for (k = 0; k < NUM_K; k = k + 1)
                for (r = 0; r < POOL_H; r = r + 1)
                    for (c = 0; c < POOL_W; c = c + 1) begin
                        a = exp_conv[k*CONV_W*CONV_H + (2*r)*CONV_W   + 2*c];
                        b = exp_conv[k*CONV_W*CONV_H + (2*r)*CONV_W   + 2*c + 1];
                        d = exp_conv[k*CONV_W*CONV_H + (2*r+1)*CONV_W + 2*c];
                        e = exp_conv[k*CONV_W*CONV_H + (2*r+1)*CONV_W + 2*c + 1];
                        m = a;
                        if (b > m) m = b;
                        if (d > m) m = d;
                        if (e > m) m = e;
                        exp_pool[k*POOL_W*POOL_H + r*POOL_W + c] = m;
                        if (m < 0) m = 0;
                        exp_relu[k*POOL_W*POOL_H + r*POOL_W + c] = m;
                        m = m >> OUT_SHIFT;
                        if (m > ((1 << OUT_W) - 1)) m = (1 << OUT_W) - 1;
                        exp_out[k*POOL_W*POOL_H + r*POOL_W + c] = m;
                    end
        end
    endtask

    // ------------------------------------------------------------------
    // Monitors (sample on posedge, stimulus is driven on negedge)
    // ------------------------------------------------------------------
    integer cc, pc, rc, oc;
    integer cyc, t_first, t_last;
    reg     started, done_seen;
    integer kk;

    always @(posedge clk) begin
        cyc = cyc + 1;
        if (pix_valid && !started) begin
            started = 1'b1;
            t_first = cyc;
        end
        if (conv_valid) begin
            for (kk = 0; kk < NUM_K; kk = kk + 1)
                got_conv[kk*CONV_W*CONV_H + cc] = $signed(conv_data[kk*ACC_W +: ACC_W]);
            cc = cc + 1;
        end
        if (pool_valid) begin
            for (kk = 0; kk < NUM_K; kk = kk + 1)
                got_pool[kk*POOL_W*POOL_H + pc] = $signed(pool_data[kk*ACC_W +: ACC_W]);
            pc = pc + 1;
        end
        if (relu_valid) begin
            for (kk = 0; kk < NUM_K; kk = kk + 1)
                got_relu[kk*POOL_W*POOL_H + rc] = relu_data[kk*ACC_W +: ACC_W];
            rc = rc + 1;
        end
        if (out_valid) begin
            for (kk = 0; kk < NUM_K; kk = kk + 1)
                got_out[kk*POOL_W*POOL_H + oc] = out_data[kk*OUT_W +: OUT_W];
            oc = oc + 1;
            t_last = cyc;
        end
        if (done) done_seen = 1'b1;
    end

    // ------------------------------------------------------------------
    // Printing helpers
    // which: 0 img, 1 conv, 2 pool, 3 relu, 4 final out
    // ------------------------------------------------------------------
    function integer get_val;
        input integer which;
        input integer idx;
        begin
            case (which)
                0: get_val = img[idx];
                1: get_val = got_conv[idx];
                2: get_val = got_pool[idx];
                3: get_val = got_relu[idx];
                default: get_val = got_out[idx];
            endcase
        end
    endfunction

    task print_mat;
        input integer which;
        input integer base;
        input integer w;
        input integer h;
        integer r, c;
        begin
            for (r = 0; r < h; r = r + 1) begin
                $write("  [");
                for (c = 0; c < w; c = c + 1)
                    $write("%7d", get_val(which, base + r*w + c));
                $write(" ]\n");
            end
        end
    endtask

    task print_results;
        integer k;
        begin
            $display("\n=== INPUT IMAGE (%0dx%0d) ===", IMG_W, IMG_H);
            print_mat(0, 0, IMG_W, IMG_H);
            for (k = 0; k < NUM_K; k = k + 1) begin
                $display("\n=== CONVOLUTION, kernel %0d (%0dx%0d) ===", k+1, CONV_W, CONV_H);
                print_mat(1, k*CONV_W*CONV_H, CONV_W, CONV_H);
            end
            for (k = 0; k < NUM_K; k = k + 1) begin
                $display("\n=== MAXPOOL 2x2, channel %0d (%0dx%0d) ===", k+1, POOL_W, POOL_H);
                print_mat(2, k*POOL_W*POOL_H, POOL_W, POOL_H);
            end
            for (k = 0; k < NUM_K; k = k + 1) begin
                $display("\n=== RELU, channel %0d (%0dx%0d) ===", k+1, POOL_W, POOL_H);
                print_mat(3, k*POOL_W*POOL_H, POOL_W, POOL_H);
            end
            for (k = 0; k < NUM_K; k = k + 1) begin
                $display("\n=== FINAL OUTPUT 8-bit unsigned (saturated), channel %0d (%0dx%0d) ===", k+1, POOL_W, POOL_H);
                print_mat(4, k*POOL_W*POOL_H, POOL_W, POOL_H);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // Stimulus
    // ------------------------------------------------------------------
    task pack_kernel;
        integer n;
        begin
            for (n = 0; n < NUM_K*K*K; n = n + 1)
                kernel_flat[n*COEF_W +: COEF_W] = kmem[n];
        end
    endtask

    task send_frame;
        input integer stall_pct;
        integer p;
        begin
            cc = 0; pc = 0; rc = 0; oc = 0;
            started = 1'b0; done_seen = 1'b0;
            for (p = 0; p < N_IMG; p = p + 1) begin
                @(negedge clk);
                while ((({$random} % 100) < stall_pct)) begin
                    pix_valid = 1'b0;
                    @(negedge clk);
                end
                pix_valid = 1'b1;
                pix_data  = img[p];
            end
            @(negedge clk);
            pix_valid = 1'b0;
            pix_data  = {DATA_W{1'b0}};
            repeat (12) @(negedge clk);
        end
    endtask

    task check_frame;
        input [8*24-1:0] name;
        integer n;
        begin
            frame_errors = 0;
            if (cc != CONV_W*CONV_H) begin $display("  [%0s] conv count %0d, expected %0d", name, cc, CONV_W*CONV_H); frame_errors = frame_errors + 1; end
            if (pc != POOL_W*POOL_H) begin $display("  [%0s] pool count %0d, expected %0d", name, pc, POOL_W*POOL_H); frame_errors = frame_errors + 1; end
            if (rc != POOL_W*POOL_H) begin $display("  [%0s] relu count %0d, expected %0d", name, rc, POOL_W*POOL_H); frame_errors = frame_errors + 1; end
            if (oc != POOL_W*POOL_H) begin $display("  [%0s] out count %0d, expected %0d", name, oc, POOL_W*POOL_H); frame_errors = frame_errors + 1; end
            if (!done_seen) begin $display("  [%0s] done pulse not seen", name); frame_errors = frame_errors + 1; end
            if (cc == CONV_W*CONV_H)
                for (n = 0; n < N_CONV; n = n + 1)
                    if (got_conv[n] !== exp_conv[n]) begin
                        if (frame_errors < 10) $display("  [%0s] CONV mismatch idx %0d: got %0d exp %0d", name, n, got_conv[n], exp_conv[n]);
                        frame_errors = frame_errors + 1;
                    end
            if (oc == POOL_W*POOL_H)
                for (n = 0; n < N_POOL; n = n + 1) begin
                    if (got_pool[n] !== exp_pool[n]) begin
                        if (frame_errors < 10) $display("  [%0s] POOL mismatch idx %0d: got %0d exp %0d", name, n, got_pool[n], exp_pool[n]);
                        frame_errors = frame_errors + 1;
                    end
                    if (got_relu[n] !== exp_relu[n]) begin
                        if (frame_errors < 10) $display("  [%0s] RELU mismatch idx %0d: got %0d exp %0d", name, n, got_relu[n], exp_relu[n]);
                        frame_errors = frame_errors + 1;
                    end
                    if (got_out[n] !== exp_out[n]) begin
                        if (frame_errors < 10) $display("  [%0s] OUT  mismatch idx %0d: got %0d exp %0d", name, n, got_out[n], exp_out[n]);
                        frame_errors = frame_errors + 1;
                    end
                end
            total_errors = total_errors + frame_errors;
            if (frame_errors == 0) $display("  [%0s] PASS (conv %0d, pool/relu/out %0d values per channel group checked)", name, N_CONV, N_POOL);
            else                   $display("  [%0s] FAIL (%0d errors)", name, frame_errors);
        end
    endtask

    integer n, t, seed_dummy;

    initial begin
        $dumpfile("sim/tb_conv_pool_relu_top.vcd");
        $dumpvars(0, tb_conv_pool_relu_top);

        total_errors = 0;
        cyc = 0; t_first = 0; t_last = 0;
        cc = 0; pc = 0; rc = 0; oc = 0;
        started = 1'b0; done_seen = 1'b0;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // ---------------- Test 1: assignment vector ----------------
        $display("\n########## TEST 1: main vector from assignment (no stall) ##########");
        $readmemh("tb/vectors/image.mem",  img);
        $readmemh("tb/vectors/kernel.mem", kmem);
        pack_kernel;
        compute_expected;
        send_frame(0);
        print_results;
        $display("\n--- Check ---");
        check_frame("main vector");
        $display("  Latency: first pixel (cycle %0d) -> last output (cycle %0d) = %0d clock cycles",
                 t_first, t_last, t_last - t_first + 1);

        // ---------------- Test 2: corner case ----------------
        $display("\n########## TEST 2: corner case (all pixels 255, extreme kernels) ##########");
        for (n = 0; n < N_IMG; n = n + 1) img[n] = 8'hFF;
        for (n = 0; n < K*K; n = n + 1) begin
            kmem[n]         = 8'h7F;   // kernel 1: all +127
            kmem[K*K + n]   = 8'h80;   // kernel 2: all -128
            kmem[2*K*K + n] = (n % 2) ? 8'h01 : 8'h00;  // kernel 3: small positive
        end
        pack_kernel;
        compute_expected;
        send_frame(0);
        check_frame("corner 255/127/-128");
        $display("  conv extremes: ch1 = %0d, ch2 = %0d (accumulator width %0d bits)",
                 got_conv[0], got_conv[CONV_W*CONV_H], ACC_W);

        // ---------------- Test 3: random frames ----------------
        $display("\n########## TEST 3: random frames (with and without input stalls) ##########");
        for (t = 0; t < 8; t = t + 1) begin
            for (n = 0; n < N_IMG; n = n + 1)       img[n]  = $random;
            for (n = 0; n < NUM_K*K*K; n = n + 1)   kmem[n] = $random;
            pack_kernel;
            compute_expected;
            send_frame((t < 3) ? 0 : 35);
            check_frame((t < 3) ? "random, no stall" : "random, 35% stall");
        end

        $display("\n==================================================");
        if (total_errors == 0) $display("ALL TESTS PASSED");
        else                   $display("TESTS FAILED: %0d errors", total_errors);
        $display("==================================================\n");
        $finish;
    end

    // Safety timeout
    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
