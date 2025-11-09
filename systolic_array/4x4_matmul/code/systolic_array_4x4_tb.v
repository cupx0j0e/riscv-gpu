`timescale 1ns/1ps

module systolic_array_4x4_tb;

    reg clk, rst;
    reg [7:0] a1, a2, a3, a4;
    reg [7:0] b1, b2, b3, b4;
    wire [31:0] c11, c12, c13, c14;
    wire [31:0] c21, c22, c23, c24;
    wire [31:0] c31, c32, c33, c34;
    wire [31:0] c41, c42, c43, c44;

    systolic_array_4x4 dut(
        .clk(clk), .rst(rst),
        .a1(a1), .a2(a2), .a3(a3), .a4(a4),
        .b1(b1), .b2(b2), .b3(b3), .b4(b4),
        .c11(c11), .c12(c12), .c13(c13), .c14(c14),
        .c21(c21), .c22(c22), .c23(c23), .c24(c24),
        .c31(c31), .c32(c32), .c33(c33), .c34(c34),
        .c41(c41), .c42(c42), .c43(c43), .c44(c44)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin

        rst = 1;

        a1=0; a2=0; a3=0; a4=0;
        b1=0; b2=0; b3=0; b4=0;
        
        #10 rst = 0;
        #15;

	// cycle 1
	@(posedge clk);
	a1 = 8'd1; a2 = 8'd2; a3 = 8'd3; a4 = 8'd4;
	b1 = 8'd5; b2 = 8'd6; b3 = 8'd7; b4 = 8'd8;

	// cycle 2
	@(posedge clk);
	a1 = 8'd2; a2 = 8'd3; a3 = 8'd4; a4 = 8'd5;
	b1 = 8'd6; b2 = 8'd7; b3 = 8'd8; b4 = 8'd9;

	// cycle 3
	@(posedge clk);
	a1 = 8'd3; a2 = 8'd4; a3 = 8'd5; a4 = 8'd6;
	b1 = 8'd7; b2 = 8'd8; b3 = 8'd9; b4 = 8'd10;

	// cycle 4
	@(posedge clk);
	a1 = 8'd4; a2 = 8'd5; a3 = 8'd6; a4 = 8'd7;
	b1 = 8'd8; b2 = 8'd9; b3 = 8'd10; b4 = 8'd11;

	// flush out values
	@(posedge clk);
        a1=0; a2=0; a3=0; a4=0;
        b1=0; b2=0; b3=0; b4=0;
            
        $display("\n========================================");
        $display("Final Results at T=%0t ns:", $time);
        $display("========================================");
        $display("%0d %0d %0d %0d", c11, c12, c13, c14);
        $display("%0d %0d %0d %0d", c21, c22, c23, c24);
        $display("%0d %0d %0d %0d", c31, c32, c33, c34);
        $display("%0d %0d %0d %0d", c41, c42, c43, c44);
        $display("========================================");
        
        if 
        (c11 == 70 && c12 == 80 && c13 == 90 && c14 == 100 
        && c21 == 96 && c22 == 110 && c23 == 124 && c24 == 138
        && c31 == 122 && c32 == 140 && c33 == 158 && c34 == 176
        && c41 == 148 && c42 == 170 && c43 == 192 && c44 == 214) begin
            $display("PASSED!");
        end else begin
            $display("FAILED!");
        end
        $display("========================================\n");
        #20; 
        
        $finish;
    end
        
endmodule

