`timescale 1ns / 1ps

module systolic_array_4x4 (
    input clk,
    input rst,
    input clear,
    input signed [7:0] a1, a2, a3, a4,
    input signed [7:0] b1, b2, b3, b4,
    output signed [31:0] c11, c12, c13, c14,
    output signed [31:0] c21, c22, c23, c24,
    output signed [31:0] c31, c32, c33, c34,
    output signed [31:0] c41, c42, c43, c44
);

    wire signed [7:0] a11_to_12, a12_to_13, a13_to_14;
    wire signed [7:0] b11_to_21, b21_to_31, b31_to_41;
    wire signed [7:0] a21_to_22, a22_to_23, a23_to_24;
    wire signed [7:0] b12_to_22, b22_to_32, b32_to_42;
    wire signed [7:0] a31_to_32, a32_to_33, a33_to_34;
    wire signed [7:0] b13_to_23, b23_to_33, b33_to_43;
    wire signed [7:0] a41_to_42, a42_to_43, a43_to_44;
    wire signed [7:0] b14_to_24, b24_to_34, b34_to_44;

    //1
    pe PE11(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a1), .b_in(b1),
        .a_out(a11_to_12), .b_out(b11_to_21),
        .c(c11));


    //2
    pe PE12(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a11_to_12), .b_in(b2), 
        .a_out(a12_to_13), .b_out(b12_to_22),
        .c(c12));
        
    pe PE21(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a2), .b_in(b11_to_21), 
        .a_out(a21_to_22), .b_out(b21_to_31),
        .c(c21));


    //3
    pe PE13(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a12_to_13), .b_in(b3), 
        .a_out(a13_to_14), .b_out(b13_to_23),
        .c(c13));
        
    pe PE22(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a21_to_22), .b_in(b12_to_22), 
        .a_out(a22_to_23), .b_out(b22_to_32),
        .c(c22));
        
    pe PE31(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a3), .b_in(b21_to_31), 
        .a_out(a31_to_32), .b_out(b31_to_41),
        .c(c31));
      
        
    //4    
    pe PE14(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a13_to_14), .b_in(b4), 
        .a_out(), .b_out(b14_to_24),
        .c(c14));


    pe PE23(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a22_to_23), .b_in(b13_to_23), 
        .a_out(a23_to_24), .b_out(b23_to_33),
        .c(c23));
        
    pe PE32(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a31_to_32), .b_in(b22_to_32), 
        .a_out(a32_to_33), .b_out(b32_to_42),
        .c(c32));

    pe PE41(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a4), .b_in(b31_to_41), 
        .a_out(a41_to_42), .b_out(),
        .c(c41));
        
        
    //5    
    pe PE24(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a23_to_24), .b_in(b14_to_24), 
        .a_out(), .b_out(b24_to_34),
        .c(c24));
        
    pe PE33(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a32_to_33), .b_in(b23_to_33), 
        .a_out(a33_to_34), .b_out(b33_to_43),
        .c(c33));
        
   pe PE42(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a41_to_42), .b_in(b32_to_42), 
        .a_out(a42_to_43), .b_out(),
        .c(c42));
        
        
        
    //6    
    pe PE34(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a33_to_34), .b_in(b24_to_34), 
        .a_out(), .b_out(b34_to_44),
        .c(c34));

    pe PE43(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a42_to_43), .b_in(b33_to_43), 
        .a_out(a43_to_44), .b_out(),
        .c(c43));


    //7	
    pe PE44(.clk(clk), .rst(rst), .clear(clear),
        .a_in(a43_to_44), .b_in(b34_to_44), 
        .a_out(), .b_out(),
        .c(c44));

endmodule

