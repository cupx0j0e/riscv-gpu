`timescale 1ns/1ps

module sys_ctrl_user #(
  parameter integer C_S_AXI_DATA_WIDTH = 32,
  parameter integer C_S_AXI_ADDR_WIDTH = 4,
  parameter integer BRAM_ADDR_WIDTH    = 11  // 2K words by default
)(
  // AXI4-Lite
  input  wire                        s_axi_aclk,
  input  wire                        s_axi_aresetn,
  input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
  input  wire                        s_axi_awvalid,
  output wire                        s_axi_awready,
  input  wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
  input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
  input  wire                        s_axi_wvalid,
  output wire                        s_axi_wready,
  output wire [1:0]                  s_axi_bresp,
  output wire                        s_axi_bvalid,
  input  wire                        s_axi_bready,
  input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
  input  wire                        s_axi_arvalid,
  output wire                        s_axi_arready,
  output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
  output wire [1:0]                  s_axi_rresp,
  output wire                        s_axi_rvalid,
  input  wire                        s_axi_rready,
  // BRAM A Port B (read)
  output reg  [BRAM_ADDR_WIDTH-1:0]  bram_a_addrb,
  output wire                        bram_a_enb,
  input  wire [31:0]                 bram_a_doutb,
  // BRAM B Port B (read)
  output reg  [BRAM_ADDR_WIDTH-1:0]  bram_b_addrb,
  output wire                        bram_b_enb,
  input  wire [31:0]                 bram_b_doutb,
  // BRAM C Port B (write)
  output reg  [BRAM_ADDR_WIDTH-1:0]  bram_c_addrb,
  output wire                        bram_c_enb,
  output wire [3:0]                  bram_c_web,
  output reg  [31:0]                 bram_c_dinb,
  // Optional status/irq
  output wire                        done
);
  // AXI-lite simple 4-reg implementation
  localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
  localparam integer OPT_MEM_ADDR_BITS = 1;
  reg axi_awready, axi_wready, axi_bvalid;
  reg axi_arready, axi_rvalid;
  reg [1:0] axi_bresp, axi_rresp;
  reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr, axi_araddr;
  reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg0, slv_reg1, slv_reg2, slv_reg3;
  wire slv_reg_wren = s_axi_wvalid && s_axi_awvalid && axi_awready && axi_wready;
  wire [1:0] reg_sel_w = s_axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB];
  wire [1:0] reg_sel_r = axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB];

  assign s_axi_awready = axi_awready;
  assign s_axi_wready  = axi_wready;
  assign s_axi_bvalid  = axi_bvalid;
  assign s_axi_bresp   = axi_bresp;
  assign s_axi_arready = axi_arready;
  assign s_axi_rvalid  = axi_rvalid;
  assign s_axi_rresp   = axi_rresp;
  assign s_axi_rdata   = (reg_sel_r==2'h0) ? slv_reg0 :
                         (reg_sel_r==2'h1) ? slv_reg1 :
                         (reg_sel_r==2'h2) ? slv_reg2 :
                         (reg_sel_r==2'h3) ? slv_reg3 : 32'd0;

  // AXI write handshake
  always @(posedge s_axi_aclk) begin
    if (!s_axi_aresetn) begin
      axi_awready <= 0; axi_wready <= 0; axi_bvalid <= 0; axi_bresp <= 0; axi_awaddr <= 0;
    end else begin
      axi_awready <= (!axi_awready && s_axi_awvalid);
      axi_wready  <= (!axi_wready  && s_axi_wvalid);
      if (slv_reg_wren) axi_awaddr <= s_axi_awaddr;
      if (slv_reg_wren) axi_bvalid <= 1'b1;
      else if (s_axi_bready) axi_bvalid <= 1'b0;
    end
  end

  // AXI read handshake
  always @(posedge s_axi_aclk) begin
    if (!s_axi_aresetn) begin
      axi_arready <= 0; axi_rvalid <= 0; axi_rresp <= 0; axi_araddr <= 0;
    end else begin
      axi_arready <= (!axi_arready && s_axi_arvalid);
      if (!axi_rvalid && s_axi_arvalid && axi_arready) begin
        axi_araddr <= s_axi_araddr;
        axi_rvalid <= 1'b1;
      end else if (axi_rvalid && s_axi_rready) axi_rvalid <= 1'b0;
    end
  end

  // Register write
  integer i;
  always @(posedge s_axi_aclk) begin
    if (!s_axi_aresetn) begin
      slv_reg0 <= 0; slv_reg1 <= 0; slv_reg2 <= 0; slv_reg3 <= 0;
    end else if (slv_reg_wren) begin
      case (reg_sel_w)
        2'h0: for (i=0;i<C_S_AXI_DATA_WIDTH/8;i=i+1) if (s_axi_wstrb[i]) slv_reg0[i*8 +:8] <= s_axi_wdata[i*8 +:8];
        2'h1: for (i=0;i<C_S_AXI_DATA_WIDTH/8;i=i+1) if (s_axi_wstrb[i]) slv_reg1[i*8 +:8] <= s_axi_wdata[i*8 +:8];
        2'h2: for (i=0;i<C_S_AXI_DATA_WIDTH/8;i=i+1) if (s_axi_wstrb[i]) slv_reg2[i*8 +:8] <= s_axi_wdata[i*8 +:8];
        2'h3: for (i=0;i<C_S_AXI_DATA_WIDTH/8;i=i+1) if (s_axi_wstrb[i]) slv_reg3[i*8 +:8] <= s_axi_wdata[i*8 +:8];
      endcase
    end else begin
      // Mirror busy/done into STATUS
      slv_reg1 <= {30'b0, busy_reg, done_reg};
    end
  end

  // Control/status and FSM
  reg start_d, done_reg, busy_reg;
  reg [BRAM_ADDR_WIDTH-1:0] a_addr, b_addr;
  reg [4:0] write_idx;
  reg [15:0] stream_cnt, flush_cnt;
  reg [2:0] state;
  localparam ST_IDLE=0, ST_CLEAR=1, ST_STREAM=2, ST_FLUSH=3, ST_WRITE=4, ST_DONE=5;
  wire start_pulse = slv_reg0[0] & ~start_d;
  wire clear_bit   = slv_reg0[1];
  wire [15:0] stream_len = (slv_reg2[15:0]!=0) ? slv_reg2[15:0] : 16'd8;
  wire [15:0] flush_len  = (slv_reg3[15:0]!=0) ? slv_reg3[15:0] : 16'd8;

  wire signed [7:0] a1 = bram_a_doutb[7:0],  a2 = bram_a_doutb[15:8],
                    a3 = bram_a_doutb[23:16],a4 = bram_a_doutb[31:24];
  wire signed [7:0] b1 = bram_b_doutb[7:0],  b2 = bram_b_doutb[15:8],
                    b3 = bram_b_doutb[23:16],b4 = bram_b_doutb[31:24];
  wire signed [31:0] c11,c12,c13,c14,c21,c22,c23,c24,c31,c32,c33,c34,c41,c42,c43,c44;
  wire [31:0] c_sel = (write_idx==0)?c11:(write_idx==1)?c12:(write_idx==2)?c13:(write_idx==3)?c14:
                      (write_idx==4)?c21:(write_idx==5)?c22:(write_idx==6)?c23:(write_idx==7)?c24:
                      (write_idx==8)?c31:(write_idx==9)?c32:(write_idx==10)?c33:(write_idx==11)?c34:
                      (write_idx==12)?c41:(write_idx==13)?c42:(write_idx==14)?c43:(write_idx==15)?c44:32'd0;

  systolic_array_4x4 u_array (
    .clk(s_axi_aclk),
    .rst(~s_axi_aresetn),
    .clear(clear_bit | (state==ST_CLEAR)),
    .a1(a1), .a2(a2), .a3(a3), .a4(a4),
    .b1(b1), .b2(b2), .b3(b3), .b4(b4),
    .c11(c11), .c12(c12), .c13(c13), .c14(c14),
    .c21(c21), .c22(c22), .c23(c23), .c24(c24),
    .c31(c31), .c32(c32), .c33(c33), .c34(c34),
    .c41(c41), .c42(c42), .c43(c43), .c44(c44)
  );

  always @(posedge s_axi_aclk) begin
    if (!s_axi_aresetn) begin
      start_d <= 0; done_reg <= 0; busy_reg <= 0;
      state <= ST_IDLE; stream_cnt <= 0; flush_cnt <= 0; write_idx <= 0;
      a_addr <= 0; b_addr <= 0; bram_c_addrb <= 0; bram_c_dinb <= 0;
    end else begin
      start_d <= slv_reg0[0];
      case (state)
        ST_IDLE: begin
          done_reg <= 0; busy_reg <= 0; stream_cnt <= 0; flush_cnt <= 0; write_idx <= 0;
          if (start_pulse) begin
            busy_reg <= 1;
            a_addr <= 0; b_addr <= 0; bram_c_addrb <= 0;
            state <= ST_CLEAR;
          end
        end
        ST_CLEAR: state <= ST_STREAM;
        ST_STREAM: begin
          if (stream_cnt >= stream_len-1) begin
            stream_cnt <= 0; state <= ST_FLUSH;
          end else begin
            stream_cnt <= stream_cnt + 1;
            a_addr <= a_addr + 1; b_addr <= b_addr + 1;
          end
        end
        ST_FLUSH: begin
          if (flush_cnt >= flush_len-1) begin
            flush_cnt <= 0; state <= ST_WRITE;
          end else flush_cnt <= flush_cnt + 1;
        end
        ST_WRITE: begin
          bram_c_dinb <= c_sel;
          bram_c_addrb <= write_idx[BRAM_ADDR_WIDTH-1:0];
          if (write_idx == 5'd15) begin
            write_idx <= 0; state <= ST_DONE;
          end else write_idx <= write_idx + 1;
        end
        ST_DONE: begin
          done_reg <= 1; busy_reg <= 0;
          if (start_pulse) begin
            done_reg <= 0; busy_reg <= 1;
            a_addr <= 0; b_addr <= 0; bram_c_addrb <= 0; write_idx <= 0;
            state <= ST_CLEAR;
          end
        end
      endcase
    end
  end

  assign bram_a_addrb = a_addr;
  assign bram_b_addrb = b_addr;
  assign bram_a_enb   = (state==ST_STREAM);
  assign bram_b_enb   = (state==ST_STREAM);
  assign bram_c_enb   = (state==ST_WRITE);
  assign bram_c_web   = (state==ST_WRITE) ? 4'hF : 4'h0;
  assign done         = done_reg;
endmodule
