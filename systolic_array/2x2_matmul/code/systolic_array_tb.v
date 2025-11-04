`timescale 1ns / 1ps

/*
 * =============================================================================
 * Matrix A:          Matrix B:          Result C:
 * [a11 a12]    ×    [b11 b12]    =    [c11 c12]
 * [a21 a22]         [b21 b22]         [c21 c22]
 *
 * cx = ax × bx 
 * =============================================================================
 */

module systolic_array_tb;
    reg clk, rst;
    reg [7:0] a1, a2;  
    reg [7:0] b1, b2; 
    wire [15:0] c11, c12, c21, c22;

    systolic_array dut (
        .clk(clk),
        .rst(rst),
        .a1(a1),  
        .a2(a2),
        .b1(b1),   
        .b2(b2),
        .c11(c11), 
        .c12(c12),
        .c21(c21),
        .c22(c22)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin

        rst = 1;
        a1 = 0; a2 = 0;
        b1 = 0; b2 = 0;
        
        #10 rst = 0;
        
        /*
         * =====================================================================
         * A = [1 2]    B = [5 6]    Expected C = [19 22]
         *     [3 4]        [7 8]                 [43 50]
         * =====================================================================
         */
        $display("\n========================================");
        $display("Starting Matrix Multiplication Test");
        $display("A = [1 2; 3 4]  B = [5 6; 7 8]");
        $display("Expected C = [19 22; 43 50]");
        $display("========================================\n");
        
        // Cycle 0: 
        #10;
        a1 <= 8'd1;  // a11
        b1 <= 8'd5;  // b11
        
        // Cycle 1:
        #10;
        a1 <= 8'd2;  // a12
        a2 <= 8'd3;  // a21
        b1 <= 8'd7;  // b21
        b2 <= 8'd6;  // b12
        
        // Cycle 2: 
        #10;
        a1 <= 8'd0;  // Done with row 1
        a2 <= 8'd4;  // a22
        b1 <= 8'd0;  // Done with column 1
        b2 <= 8'd8;  // b22
        
        // Flush pipeline
        #10;  
        a1 <= 8'd0;
        a2 <= 8'd0;
        b1 <= 8'd0;
        b2 <= 8'd0;
        
        repeat(10) @(posedge clk);
        
        // results
        $display("\n========================================");
        $display("Final Results at T=%0t ns:", $time);
        $display("========================================");
        $display("C11 = %0d  (Expected: 19)", c11);
        $display("C12 = %0d  (Expected: 22)", c12);
        $display("C21 = %0d  (Expected: 43)", c21);
        $display("C22 = %0d  (Expected: 50)", c22);
        $display("========================================");
        
        // Verification
        if (c11 == 19 && c12 == 22 && c21 == 43 && c22 == 50) begin
            $display("PASSED!");
        end else begin
            $display("FAILED!");
        end
        $display("========================================\n");
        #20; 
        
        $finish;
    end
    
    initial begin
        $display("\n Time | rst | a1  a2 | b1  b2 | c11  c12  c21  c22");
        $display("------|-----|--------|--------|-------------------------");
        $monitor("%5t |  %b  | %2d  %2d | %2d  %2d | %3d  %3d  %3d  %3d", 
                 $time, rst, a1, a2, b1, b2, c11, c12, c21, c22);
    end
    
endmodule
