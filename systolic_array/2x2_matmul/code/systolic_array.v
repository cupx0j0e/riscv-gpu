`timescale 1ns / 1ps

module systolic_array (
    input clk,
    input rst,
    input [7:0] a1, a2,   
    input [7:0] b1, b2,  
    output [15:0] c11, c12, c21, c22
);

    wire [7:0] a11_to_12, a21_to_22;
    wire [7:0] b11_to_21, b12_to_22;

    // Row 1, Column 1
    pe pe11 (
        .clk(clk), .rst(rst),
        .a_in(a1), .b_in(b1),
        .a_out(a11_to_12), .b_out(b11_to_21),
        .c(c11)
    );

    // Row 1, Column 2
    pe pe12 (
        .clk(clk), .rst(rst),
        .a_in(a11_to_12), .b_in(b2),
        .a_out(), .b_out(b12_to_22),
        .c(c12)
    );

    // Row 2, Column 1
    pe pe21 (
        .clk(clk), .rst(rst),
        .a_in(a2), .b_in(b11_to_21),
        .a_out(a21_to_22), .b_out(),
        .c(c21)
    );

    // Row 2, Column 2
    pe pe22 (
        .clk(clk), .rst(rst),
        .a_in(a21_to_22), .b_in(b12_to_22),
        .a_out(), .b_out(),
        .c(c22)
    );

endmodule

