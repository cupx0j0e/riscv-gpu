`timescale 1ns/1ps

module tb_NxN_systolic_array_8x8;

    // ================= PARAMETERS =================
    parameter integer N  = 8;
    parameter integer DW = 8;
    parameter integer CW = 32;

    // ================= SIGNALS =================
    reg  clk, rst, clear;
    reg  signed [DW-1:0] a_in [0:N-1];
    reg  signed [DW-1:0] b_in [0:N-1];
    wire signed [CW-1:0] c_out [0:N-1][0:N-1];

    integer A     [0:N-1][0:N-1];
    integer B     [0:N-1][0:N-1];
    integer C_ref [0:N-1][0:N-1];

    integer i, j, k, t;
    integer errors;
    integer cycle;

    // ================= DUT =================
    systolic_array #(
        .N (N),
        .DW(DW),
        .CW(CW)
    ) dut (
        .clk   (clk),
        .rst   (rst),
        .clear (clear),
        .a_in  (a_in),
        .b_in  (b_in),
        .c_out (c_out)
    );

    // ================= CLOCK =================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ================= DUMP =================
    initial begin
        $dumpfile("8x8.vcd");
        $dumpvars(0, tb_NxN_systolic_array_8x8);
    end

    // ================= MONITOR =================
    integer r, c;
    always @(posedge clk) begin
        cycle = cycle + 1;

        $display("\n==============================");
        $display("Cycle %0d", cycle);
        $display("==============================");

        $display("Inputs used by PEs:");
        for (i = 0; i < N; i = i + 1)
            $display("  A_row[%0d] = %0d   B_col[%0d] = %0d",
                     i, a_in[i], i, b_in[i]);

        $display("C matrix:");
        for (r = 0; r < N; r = r + 1) begin
            for (c = 0; c < N; c = c + 1)
                $write("%6d ", c_out[r][c]);
            $write("\n");
        end
    end

    // ================= TEST =================
    initial begin
        rst    = 1;
        clear = 0;
        cycle = 0;

        for (i = 0; i < N; i = i + 1) begin
            a_in[i] = 0;
            b_in[i] = 0;
        end

        #20 rst = 0;

        // A[i][j] = i + j + 1
        // B = identity matrix
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < N; j = j + 1) begin
                A[i][j] = i + j + 1;
                B[i][j] = (i == j) ? 1 : 0;
            end

        // Golden reference
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < N; j = j + 1) begin
                C_ref[i][j] = 0;
                for (k = 0; k < N; k = k + 1)
                    C_ref[i][j] = C_ref[i][j] + A[i][k] * B[k][j];
            end

        // Clear array
        @(posedge clk);
        clear = 1;
        @(posedge clk);
        clear = 0;

        // ================= SYSTOLIC FEED =================
        for (t = 0; t < 2*N-1; t = t + 1) begin
            @(posedge clk);

            // Feed A (rightward)
            for (i = 0; i < N; i = i + 1) begin
                k = t - i;
                if (k >= 0 && k < N)
                    a_in[i] = A[i][k];
                else
                    a_in[i] = 0;
            end

            // Feed B (downward)
            for (j = 0; j < N; j = j + 1) begin
                k = t - j;
                if (k >= 0 && k < N)
                    b_in[j] = B[k][j];
                else
                    b_in[j] = 0;
            end
        end

        // Flush pipeline
        @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin
            a_in[i] = 0;
            b_in[i] = 0;
        end

        repeat (N + 5) @(posedge clk);

        // ================= RESULTS =================
        $display("\n========================================");
        $display("Final Results:");
        $display("========================================");

        for (i = 0; i < N; i = i + 1) begin
            for (j = 0; j < N; j = j + 1)
                $write("%0d ", c_out[i][j]);
            $write("\n");
        end

        $display("========================================");

        // ================= CHECK =================
        errors = 0;
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < N; j = j + 1)
                if (c_out[i][j] !== C_ref[i][j])
                    errors = errors + 1;

        if (errors == 0)
            $display("PASSED!");
        else
            $display("FAILED! mismatches = %0d", errors);

        $display("========================================\n");

        #20 $finish;
    end

endmodule

