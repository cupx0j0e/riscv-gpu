`timescale 1ns / 1ps

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
        #15;
        
        $display("\n========================================");
        $display("Starting Matrix Multiplication Test");
        $display("A = [1 2]\n    [3 4]\n\nB = [5 6]\n    [7 8]\n\n");
        $display("Expected C = [19 22]\n             [43 50]");
        $display("========================================\n");
        
        // cycle 1
        @(posedge clk);
        a1 = 8'd1; a2 = 8'd3;
        b1 = 8'd5; b2 = 8'd6;

        // cycle 2
        @(posedge clk);
        a1 = 8'd2; a2 = 8'd4;
        b1 = 8'd7; b2 = 8'd8;

        // flush out data
        @(posedge clk);
        a1 = 0; a2 = 0;
        b1 = 0; b2 = 0;

        
        $display("\n========================================");
        $display("Final Results at T=%0t ns:", $time);
        $display("========================================");
        $display("C11 = %0d  (Expected: 19)", c11);
        $display("C12 = %0d  (Expected: 22)", c12);
        $display("C21 = %0d  (Expected: 43)", c21);
        $display("C22 = %0d  (Expected: 50)", c22);
        $display("========================================");
        
        if (c11 == 19 && c12 == 22 && c21 == 43 && c22 == 50) begin
            $display("PASSED!");
        end else begin
            $display("FAILED!");
        end
        $display("========================================\n");
        #20; 
        
        $finish;
    end
    
endmodule
