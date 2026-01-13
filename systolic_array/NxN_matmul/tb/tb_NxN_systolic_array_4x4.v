`timescale 1ns/1ps

module tb_NxN_systolic_array_4x4;

    // ================= PARAMETERS =================
    parameter integer N  = 4;
    parameter integer DW = 8;
    parameter integer CW = 32;

    // ================= SIGNALS =================
    reg clk, rst, clear;

    reg  signed [DW-1:0] a_in [0:N-1];
    reg  signed [DW-1:0] b_in [0:N-1];
    wire signed [CW-1:0] c_out [0:N-1][0:N-1];

    integer A     [0:N-1][0:N-1];
    integer B     [0:N-1][0:N-1];
    integer C_ref [0:N-1][0:N-1];

    integer i, j, k, t;
    integer cycle;
    integer errors;

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
        $dumpfile("4x4.vcd");
        $dumpvars(0, tb_NxN_systolic_array_4x4);
    end

    // ================= MONITOR =================
    integer r, c;

    reg signed [DW-1:0] a_used [0:N-1];
    reg signed [DW-1:0] b_used [0:N-1];

    initial begin
        for (r = 0; r < N; r = r + 1) begin
            a_used[r] = 0;
            b_used[r] = 0;
        end
    end

    always @(posedge clk) begin
        cycle = cycle + 1;

        for (r = 0; r < N; r = r + 1) begin
            a_used[r] = a_in[r];
            b_used[r] = b_in[r];
        end

        $display("\n==============================");
        $display("Cycle %0d", cycle);
        $display("==============================");

        $display("Inputs used by PEs:");
        for (r = 0; r < N; r = r + 1)
            $display("  A_row[%0d] = %0d   B_col[%0d] = %0d",
                     r, a_used[r], r, b_used[r]);

        $display("C matrix:");
        for (r = 0; r < N; r = r + 1) begin
            for (c = 0; c < N; c = c + 1)
                $write("%6d ", c_out[r][c]);
            $write("\n");
        end
    end

    // ================= TEST =================
    initial begin
        rst   = 1;
        clear = 0;
        cycle = 0;

        for (i = 0; i < N; i = i + 1) begin
            a_in[i] = 0;
            b_in[i] = 0;
        end

        #20 rst = 0;

        // -------- Matrix A --------
        A[0][0]=1; A[0][1]=2; A[0][2]=3; A[0][3]=4;
        A[1][0]=2; A[1][1]=3; A[1][2]=4; A[1][3]=5;
        A[2][0]=3; A[2][1]=4; A[2][2]=5; A[2][3]=6;
        A[3][0]=4; A[3][1]=5; A[3][2]=6; A[3][3]=7;

        // -------- Matrix B --------
        B[0][0]=5;  B[0][1]=6;  B[0][2]=7;  B[0][3]=8;
        B[1][0]=6;  B[1][1]=7;  B[1][2]=8;  B[1][3]=9;
        B[2][0]=7;  B[2][1]=8;  B[2][2]=9;  B[2][3]=10;
        B[3][0]=8;  B[3][1]=9;  B[3][2]=10; B[3][3]=11;

        // -------- Golden reference --------
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < N; j = j + 1) begin
                C_ref[i][j] = 0;
                for (k = 0; k < N; k = k + 1)
                    C_ref[i][j] = C_ref[i][j] + A[i][k] * B[k][j];
            end

        // -------- Clear accumulators --------
        @(posedge clk);
        clear = 1;
        @(posedge clk);
        clear = 0;

        // ================= SYSTOLIC FEED =================
        for (t = 0; t < 2*N-1; t = t + 1) begin
            @(posedge clk);

            // Feed A (by row)
            for (i = 0; i < N; i = i + 1) begin
                k = t - i;
                if (k >= 0 && k < N)
                    a_in[i] = A[i][k];
                else
                    a_in[i] = 0;
            end

            // Feed B (by column)
            for (j = 0; j < N; j = j + 1) begin
                k = t - j;
                if (k >= 0 && k < N)
                    b_in[j] = B[k][j];
                else
                    b_in[j] = 0;
            end
        end

        // -------- Flush --------
        @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin
            a_in[i] = 0;
            b_in[i] = 0;
        end

        repeat (10) @(posedge clk);

        // ================= RESULTS =================
        $display("\n========================================");
        $display("Final Results:");
        $display("========================================");

        for (i = 0; i < N; i = i + 1) begin
            for (j = 0; j < N; j = j + 1)
                $write("%0d ", c_out[i][j]);
            $write("\n");
        end

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

