module systolic_array_4x4 (
    input clk,
    input rst,
    input [7:0] a1, a2, a3, a4,
    input [7:0] b1, b2, b3, b4,
    output [31:0] c11, c12, c13, c14,
    output [31:0] c21, c22, c23, c24,
    output [31:0] c31, c32, c33, c34,
    output [31:0] c41, c42, c43, c44
);

    pe PE11(.clk(clk), .rst(rst),
        .a_in(a1), .b_in(b1), .c(c11));

    pe PE12(.clk(clk), .rst(rst),
        .a_in(a1), .b_in(b2), .c(c12));

    pe PE13(.clk(clk), .rst(rst),
        .a_in(a1), .b_in(b3), .c(c13));

    pe PE14(.clk(clk), .rst(rst),
        .a_in(a1), .b_in(b4), .c(c14));



    pe PE21(.clk(clk), .rst(rst),
        .a_in(a2), .b_in(b1), .c(c21));

    pe PE22(.clk(clk), .rst(rst),
        .a_in(a2), .b_in(b2), .c(c22));

    pe PE23(.clk(clk), .rst(rst),
        .a_in(a2), .b_in(b3), .c(c23));

    pe PE24(.clk(clk), .rst(rst),
        .a_in(a2), .b_in(b4), .c(c24));



    pe PE31(.clk(clk), .rst(rst),
        .a_in(a3), .b_in(b1), .c(c31));

    pe PE32(.clk(clk), .rst(rst),
        .a_in(a3), .b_in(b2), .c(c32));

    pe PE33(.clk(clk), .rst(rst),
        .a_in(a3), .b_in(b3), .c(c33));

    pe PE34(.clk(clk), .rst(rst),
        .a_in(a3), .b_in(b4), .c(c34));



    pe PE41(.clk(clk), .rst(rst),
        .a_in(a4), .b_in(b1), .c(c41));

    pe PE42(.clk(clk), .rst(rst),
        .a_in(a4), .b_in(b2), .c(c42));

    pe PE43(.clk(clk), .rst(rst),
        .a_in(a4), .b_in(b3), .c(c43));

    pe PE44(.clk(clk), .rst(rst),
        .a_in(a4), .b_in(b4), .c(c44));

endmodule

