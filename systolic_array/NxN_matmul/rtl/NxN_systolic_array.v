`timescale 1ns/1ps

module systolic_array #(
    parameter integer N  = 4,   // array size
    parameter integer DW = 8,   // data width
    parameter integer CW = 32   // accumulator width
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    clear,

    input  wire signed [DW-1:0]     a_in  [0:N-1],   // row inputs
    input  wire signed [DW-1:0]     b_in  [0:N-1],   // column inputs

    output wire signed [CW-1:0]     c_out [0:N-1][0:N-1]
);

    // ================= INTERNAL BUSES =================
    wire signed [DW-1:0] a_bus [0:N-1][0:N];     // flows right
    wire signed [DW-1:0] b_bus [0:N][0:N-1];     // flows down

    genvar i, j;

    // ================= INPUT BOUNDARY =================
    generate
        for (i = 0; i < N; i = i + 1) begin
            assign a_bus[i][0] = a_in[i];
            assign b_bus[0][i] = b_in[i];
        end
    endgenerate

    // ================= PE GRID =================
    generate
        for (i = 0; i < N; i = i + 1) begin : ROW
            for (j = 0; j < N; j = j + 1) begin : COL
                pe PE (
                    .clk   (clk),
                    .rst   (rst),
                    .clear (clear),

                    .a_in  (a_bus[i][j]),
                    .a_out (a_bus[i][j+1]),

                    .b_in  (b_bus[i][j]),
                    .b_out (b_bus[i+1][j]),

                    .c     (c_out[i][j])
                );
            end
        end
    endgenerate

endmodule

