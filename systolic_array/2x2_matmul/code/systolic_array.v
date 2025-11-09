`timescale 1ns / 1ps

module systolic_array (
    input clk,
    input rst,
    input [7:0] a1, a2,   
    input [7:0] b1, b2,  
    output [15:0] c11, c12, c21, c22
);


    // Row 1, Column 1
    pe pe11 (
        .clk(clk), .rst(rst),
        .a_in(a1), .b_in(b1),
        .c(c11)
    );

    // Row 1, Column 2
    pe pe12 (
        .clk(clk), .rst(rst),
        .a_in(a1), .b_in(b2),
        .c(c12)
    );

    // Row 2, Column 1
    pe pe21 (
        .clk(clk), .rst(rst),
        .a_in(a2), .b_in(b1),
        .c(c21)
    );

    // Row 2, Column 2
    pe pe22 (
        .clk(clk), .rst(rst),
        .a_in(a2), .b_in(b2),
        .c(c22)
    );

endmodule

