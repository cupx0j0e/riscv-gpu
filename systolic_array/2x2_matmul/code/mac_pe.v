`timescale 1ns / 1ps
// for 2x2 matrix
module pe (
    input clk,          
    input rst,         
    input [7:0] a_in,   
    input [7:0] b_in,   
    output reg [7:0] a_out, 
    output reg [7:0] b_out, 
    output reg [15:0] c   
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out <= 0;
            b_out <= 0;
            c <= 0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            c <= c + (a_in * b_in);
        end
    end
endmodule

