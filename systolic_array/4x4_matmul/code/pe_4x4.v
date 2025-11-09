module pe (
    input clk,
    input rst,
    input [7:0] a_in,
    input [7:0] b_in,
    output reg [31:0] c
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c <= 0;
        end else begin
            c <= c + a_in * b_in; 
        end
    end
endmodule

