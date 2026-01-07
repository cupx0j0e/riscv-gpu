
`timescale 1 ns / 1 ps

	module sys_ctrl_slave_lite_v1_0_S00_AXI #
	(
		// Users to add parameters here

		// User parameters ends
		// Do not modify the parameters beyond this line

		// Width of S_AXI data bus
		parameter integer C_S_AXI_DATA_WIDTH	= 32,
		// Width of S_AXI address bus
		parameter integer C_S_AXI_ADDR_WIDTH	= 4,
		// Depth of BRAM (address width for port B)
		parameter integer BRAM_ADDR_WIDTH      = 11
	)
	(
		// Global Clock Signal
		input wire  S_AXI_ACLK,
		// Global Reset Signal. This Signal is Active LOW
		input wire  S_AXI_ARESETN,
		// Write address (issued by master, acceped by Slave)
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
		// Write channel Protection type. This signal indicates the
    		// privilege and security level of the transaction, and whether
    		// the transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_AWPROT,
		// Write address valid. This signal indicates that the master signaling
    		// valid write address and control information.
		input wire  S_AXI_AWVALID,
		// Write address ready. This signal indicates that the slave is ready
    		// to accept an address and associated control signals.
		output wire  S_AXI_AWREADY,
		// Write data (issued by master, acceped by Slave) 
		input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
		// Write strobes. This signal indicates which byte lanes hold
    		// valid data. There is one write strobe bit for each eight
    		// bits of the write data bus.    
		input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
		// Write valid. This signal indicates that valid write
    		// data and strobes are available.
		input wire  S_AXI_WVALID,
		// Write ready. This signal indicates that the slave
    		// can accept the write data.
		output wire  S_AXI_WREADY,
		// Write response. This signal indicates the status
    		// of the write transaction.
		output wire [1 : 0] S_AXI_BRESP,
		// Write response valid. This signal indicates that the channel
    		// is signaling a valid write response.
		output wire  S_AXI_BVALID,
		// Response ready. This signal indicates that the master
    		// can accept a write response.
		input wire  S_AXI_BREADY,
		// Read address (issued by master, acceped by Slave)
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
		// Protection type. This signal indicates the privilege
    		// and security level of the transaction, and whether the
    		// transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_ARPROT,
		// Read address valid. This signal indicates that the channel
    		// is signaling valid read address and control information.
		input wire  S_AXI_ARVALID,
		// Read address ready. This signal indicates that the slave is
    		// ready to accept an address and associated control signals.
		output wire  S_AXI_ARREADY,
		// Read data (issued by slave)
		output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
		// Read response. This signal indicates the status of the
    		// read transfer.
		output wire [1 : 0] S_AXI_RRESP,
    		// Read valid. This signal indicates that the channel is
    		// signaling the required read data.
		output wire  S_AXI_RVALID,
		// Read ready. This signal indicates that the master can
    		// accept the read data and response information.
		input wire  S_AXI_RREADY,

		// BRAM A Port B (read)
		output wire [BRAM_ADDR_WIDTH-1:0] bram_a_addrb,
		output wire                       bram_a_enb,
		input  wire [31:0]                bram_a_doutb,

		// BRAM B Port B (read)
		output wire [BRAM_ADDR_WIDTH-1:0] bram_b_addrb,
		output wire                       bram_b_enb,
		input  wire [31:0]                bram_b_doutb,

		// BRAM C Port B (write)
		output wire [BRAM_ADDR_WIDTH-1:0] bram_c_addrb,
		output wire                       bram_c_enb,
		output wire [3:0]                 bram_c_web,
		output wire [31:0]                bram_c_dinb,
		input  wire [31:0]                bram_c_doutb, // optional readback

		// Optional status/interrupt
		output wire                       done
	);

	// AXI4LITE signals
	reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr;
	reg  	axi_awready;
	reg  	axi_wready;
	reg [1 : 0] 	axi_bresp;
	reg  	axi_bvalid;
	reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr;
	reg  	axi_arready;
	reg [1 : 0] 	axi_rresp;
	reg  	axi_rvalid;

	// Example-specific design signals
	// local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
	// ADDR_LSB is used for addressing 32/64 bit registers/memories
	// ADDR_LSB = 2 for 32 bits (n downto 2)
	// ADDR_LSB = 3 for 64 bits (n downto 3)
	localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
	localparam integer OPT_MEM_ADDR_BITS = 1;
	//----------------------------------------------
	//-- Signals for user logic register space example
	//------------------------------------------------
	//-- Number of Slave Registers 4
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg0;
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg1;
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg2;
	reg [C_S_AXI_DATA_WIDTH-1:0]	slv_reg3;
	integer	 byte_index;

	// User control/status and BRAM/array signals
	reg start_d;
	reg done_reg;
	reg busy_reg;
	reg [BRAM_ADDR_WIDTH-1:0] bram_a_addrb_r;
	reg [BRAM_ADDR_WIDTH-1:0] bram_b_addrb_r;
	reg [4:0]                 write_idx;
	reg [15:0]                stream_cnt;
	reg [15:0]                flush_cnt;
	reg [2:0]                 ctrl_state;

	localparam ST_IDLE   = 3'd0;
	localparam ST_CLEAR  = 3'd1;
	localparam ST_STREAM = 3'd2;
	localparam ST_FLUSH  = 3'd3;
	localparam ST_WRITE  = 3'd4;
	localparam ST_DONE   = 3'd5;

	wire start_pulse = slv_reg0[0] & ~start_d;
	wire clear_bit   = slv_reg0[1];

	wire [15:0] stream_len = (slv_reg2[15:0] != 16'd0) ? slv_reg2[15:0] : 16'd8;
	wire [15:0] flush_len  = (slv_reg3[15:0] != 16'd0) ? slv_reg3[15:0] : 16'd8;

	wire signed [7:0] a1 = bram_a_doutb[7:0];
	wire signed [7:0] a2 = bram_a_doutb[15:8];
	wire signed [7:0] a3 = bram_a_doutb[23:16];
	wire signed [7:0] a4 = bram_a_doutb[31:24];

	wire signed [7:0] b1 = bram_b_doutb[7:0];
	wire signed [7:0] b2 = bram_b_doutb[15:8];
	wire signed [7:0] b3 = bram_b_doutb[23:16];
	wire signed [7:0] b4 = bram_b_doutb[31:24];

	wire signed [31:0] c11, c12, c13, c14;
	wire signed [31:0] c21, c22, c23, c24;
	wire signed [31:0] c31, c32, c33, c34;
	wire signed [31:0] c41, c42, c43, c44;

	wire [31:0] c_word_sel_nxt = (write_idx == 5'd0)  ? c11 :
	                             (write_idx == 5'd1)  ? c12 :
	                             (write_idx == 5'd2)  ? c13 :
	                             (write_idx == 5'd3)  ? c14 :
	                             (write_idx == 5'd4)  ? c21 :
	                             (write_idx == 5'd5)  ? c22 :
	                             (write_idx == 5'd6)  ? c23 :
	                             (write_idx == 5'd7)  ? c24 :
	                             (write_idx == 5'd8)  ? c31 :
	                             (write_idx == 5'd9)  ? c32 :
	                             (write_idx == 5'd10) ? c33 :
	                             (write_idx == 5'd11) ? c34 :
	                             (write_idx == 5'd12) ? c41 :
	                             (write_idx == 5'd13) ? c42 :
	                             (write_idx == 5'd14) ? c43 :
	                             (write_idx == 5'd15) ? c44 : 32'd0;

	// I/O Connections assignments

	assign S_AXI_AWREADY	= axi_awready;
	assign S_AXI_WREADY	= axi_wready;
	assign S_AXI_BRESP	= axi_bresp;
	assign S_AXI_BVALID	= axi_bvalid;
	assign S_AXI_ARREADY	= axi_arready;
	assign S_AXI_RRESP	= axi_rresp;
	assign S_AXI_RVALID	= axi_rvalid;
	 //state machine varibles 
	 reg [1:0] state_write;
	 reg [1:0] state_read;
	 //State machine local parameters
	 localparam Idle = 2'b00,Raddr = 2'b10,Rdata = 2'b11 ,Waddr = 2'b10,Wdata = 2'b11;
	// Implement Write state machine
	// Outstanding write transactions are not supported by the slave i.e., master should assert bready to receive response on or before it starts sending the new transaction
	always @(posedge S_AXI_ACLK)                                 
	  begin                                 
	     if (S_AXI_ARESETN == 1'b0)                                 
	       begin                                 
	         axi_awready <= 0;                                 
	         axi_wready <= 0;                                 
	         axi_bvalid <= 0;                                 
	         axi_bresp <= 0;                                 
	         axi_awaddr <= 0;                                 
	         state_write <= Idle;                                 
	       end                                 
	     else                                  
	       begin                                 
	         case(state_write)                                 
	           Idle:                                      
	             begin                                 
	               if(S_AXI_ARESETN == 1'b1)                                  
	                 begin                                 
	                   axi_awready <= 1'b1;                                 
	                   axi_wready <= 1'b1;                                 
	                   state_write <= Waddr;                                 
	                 end                                 
	               else state_write <= state_write;                                 
	             end                                 
	           Waddr:        //At this state, slave is ready to receive address along with corresponding control signals and first data packet. Response valid is also handled at this state                                 
	             begin                                 
	               if (S_AXI_AWVALID && S_AXI_AWREADY)                                 
	                  begin                                 
	                    axi_awaddr <= S_AXI_AWADDR;                                 
	                    if(S_AXI_WVALID)                                  
	                      begin                                   
	                        axi_awready <= 1'b1;                                 
	                        state_write <= Waddr;                                 
	                        axi_bvalid <= 1'b1;                                 
	                      end                                 
	                    else                                  
	                      begin                                 
	                        axi_awready <= 1'b0;                                 
	                        state_write <= Wdata;                                 
	                        if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;                                 
	                      end                                 
	                  end                                 
	               else                                  
	                  begin                                 
	                    state_write <= state_write;                                 
	                    if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;                                 
	                   end                                 
	             end                                 
	          Wdata:        //At this state, slave is ready to receive the data packets until the number of transfers is equal to burst length                                 
	             begin                                 
	               if (S_AXI_WVALID)                                 
	                 begin                                 
	                   state_write <= Waddr;                                 
	                   axi_bvalid <= 1'b1;                                 
	                   axi_awready <= 1'b1;                                 
	                 end                                 
	                else                                  
	                 begin                                 
	                   state_write <= state_write;                                 
	                   if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;                                 
	                 end                                              
	             end                                 
	          endcase                                 
	        end                                 
	      end                                 

	// Implement memory mapped register select and write logic generation
	// The write data is accepted and written to memory mapped registers when
	// axi_awready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted. Write strobes are used to
	// select byte enables of slave registers while writing.
	// These registers are cleared when reset (active low) is applied.
	// Slave register write enable is asserted when valid address and data are available
	// and the slave is ready to accept the write address and write data.
	 

	always @( posedge S_AXI_ACLK )
	begin
	  if ( S_AXI_ARESETN == 1'b0 )
	    begin
	      slv_reg0 <= 0;
	      slv_reg1 <= 0;
	      slv_reg2 <= 0;
	      slv_reg3 <= 0;
	    end 
	  else begin
	    if (S_AXI_WVALID)
	      begin
	        case ( (S_AXI_AWVALID) ? S_AXI_AWADDR[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] : axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
	          2'h0:
	            for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXI_WSTRB[byte_index] == 1 ) begin
	                // CTRL register
	                slv_reg0[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
	              end  
	          2'h1: begin
	                // STATUS is read-only; ignore writes
	                slv_reg1 <= slv_reg1;
	              end
	          2'h2:
	            for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXI_WSTRB[byte_index] == 1 ) begin
	                // STREAM_LEN
	                slv_reg2[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
	              end  
	          2'h3:
	            for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
	              if ( S_AXI_WSTRB[byte_index] == 1 ) begin
	                // FLUSH_LEN
	                slv_reg3[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
	              end  
	          default : begin
	                      slv_reg0 <= slv_reg0;
	                      slv_reg1 <= slv_reg1;
	                      slv_reg2 <= slv_reg2;
	                      slv_reg3 <= slv_reg3;
	                    end
	        endcase
	      end else begin
	        slv_reg1 <= {30'b0, busy_reg, done_reg};
	      end
	  end
	end    

	// Implement read state machine
	  always @(posedge S_AXI_ACLK)                                       
	    begin                                       
	      if (S_AXI_ARESETN == 1'b0)                                       
	        begin                                       
	         //asserting initial values to all 0's during reset                                       
	         axi_arready <= 1'b0;                                       
	         axi_rvalid <= 1'b0;                                       
	         axi_rresp <= 1'b0;                                       
	         state_read <= Idle;                                       
	        end                                       
	      else                                       
	        begin                                       
	          case(state_read)                                       
	            Idle:     //Initial state inidicating reset is done and ready to receive read/write transactions                                       
	              begin                                                
	                if (S_AXI_ARESETN == 1'b1)                                        
	                  begin                                       
	                    state_read <= Raddr;                                       
	                    axi_arready <= 1'b1;                                       
	                  end                                       
	                else state_read <= state_read;                                       
	              end                                       
	            Raddr:        //At this state, slave is ready to receive address along with corresponding control signals                                       
	              begin                                       
	                if (S_AXI_ARVALID && S_AXI_ARREADY)                                       
	                  begin                                       
	                    state_read <= Rdata;                                       
	                    axi_araddr <= S_AXI_ARADDR;                                       
	                    axi_rvalid <= 1'b1;                                       
	                    axi_arready <= 1'b0;                                       
	                  end                                       
	                else state_read <= state_read;                                       
	              end                                       
	            Rdata:        //At this state, slave is ready to send the data packets until the number of transfers is equal to burst length                                       
	              begin                                           
	                if (S_AXI_RVALID && S_AXI_RREADY)                                       
	                  begin                                       
	                    axi_rvalid <= 1'b0;                                       
	                    axi_arready <= 1'b1;                                       
	                    state_read <= Raddr;                                       
	                  end                                       
	                else state_read <= state_read;                                       
	              end                                       
	           endcase                                       
	          end                                       
	        end                                         
	// Implement memory mapped register select and read logic generation
	  assign S_AXI_RDATA = (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 2'h0) ? slv_reg0 : (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 2'h1) ? slv_reg1 : (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 2'h2) ? slv_reg2 : (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 2'h3) ? slv_reg3 : 0; 
	// Add user logic here

	// Systolic array instance
	systolic_array_4x4 u_array (
		.clk   (S_AXI_ACLK),
		.rst   (~S_AXI_ARESETN),
		.clear (clear_bit | (ctrl_state == ST_CLEAR)),
		.a1(a1), .a2(a2), .a3(a3), .a4(a4),
		.b1(b1), .b2(b2), .b3(b3), .b4(b4),
		.c11(c11), .c12(c12), .c13(c13), .c14(c14),
		.c21(c21), .c22(c22), .c23(c23), .c24(c24),
		.c31(c31), .c32(c32), .c33(c33), .c34(c34),
		.c41(c41), .c42(c42), .c43(c43), .c44(c44)
	);

	// Control FSM: start/stream/flush/write/done
	always @(posedge S_AXI_ACLK) begin
		if (S_AXI_ARESETN == 1'b0) begin
			start_d        <= 1'b0;
			done_reg       <= 1'b0;
			busy_reg       <= 1'b0;
			bram_a_addrb_r <= {BRAM_ADDR_WIDTH{1'b0}};
			bram_b_addrb_r <= {BRAM_ADDR_WIDTH{1'b0}};
			write_idx      <= 5'd0;
			stream_cnt     <= 16'd0;
			flush_cnt      <= 16'd0;
			ctrl_state     <= ST_IDLE;
		end else begin
			start_d <= slv_reg0[0];
			case (ctrl_state)
				ST_IDLE: begin
					done_reg   <= 1'b0;
					busy_reg   <= 1'b0;
					stream_cnt <= 16'd0;
					flush_cnt  <= 16'd0;
					write_idx  <= 5'd0;
					if (start_pulse) begin
						busy_reg       <= 1'b1;
						bram_a_addrb_r <= {BRAM_ADDR_WIDTH{1'b0}};
						bram_b_addrb_r <= {BRAM_ADDR_WIDTH{1'b0}};
						ctrl_state     <= ST_CLEAR;
					end
				end
				ST_CLEAR: begin
					ctrl_state <= ST_STREAM;
				end
				ST_STREAM: begin
					if (stream_cnt >= stream_len - 1) begin
						stream_cnt <= 16'd0;
						ctrl_state <= ST_FLUSH;
					end else begin
						stream_cnt     <= stream_cnt + 1'b1;
						bram_a_addrb_r <= bram_a_addrb_r + 1'b1;
						bram_b_addrb_r <= bram_b_addrb_r + 1'b1;
					end
				end
				ST_FLUSH: begin
					if (flush_cnt >= flush_len - 1) begin
						flush_cnt  <= 16'd0;
						ctrl_state <= ST_WRITE;
					end else begin
						flush_cnt <= flush_cnt + 1'b1;
					end
				end
				ST_WRITE: begin
					if (write_idx >= 5'd15) begin
						write_idx  <= 5'd0;
						ctrl_state <= ST_DONE;
					end else begin
						write_idx <= write_idx + 1'b1;
					end
				end
				ST_DONE: begin
					done_reg <= 1'b1;
					busy_reg <= 1'b0;
					if (start_pulse) begin
						done_reg       <= 1'b0;
						busy_reg       <= 1'b1;
						bram_a_addrb_r <= {BRAM_ADDR_WIDTH{1'b0}};
						bram_b_addrb_r <= {BRAM_ADDR_WIDTH{1'b0}};
						write_idx      <= 5'd0;
						ctrl_state     <= ST_CLEAR;
					end
				end
				default: ctrl_state <= ST_IDLE;
			endcase
		end
	end

  // BRAM controllers use byte addressing; shift the word index by 2 to land on 32-bit words.
  assign bram_a_addrb = bram_a_addrb_r << 2;
  assign bram_b_addrb = bram_b_addrb_r << 2;
  assign bram_c_addrb = {{(BRAM_ADDR_WIDTH-5){1'b0}}, write_idx} << 2;

	assign bram_a_enb = (ctrl_state == ST_STREAM);
	assign bram_b_enb = (ctrl_state == ST_STREAM);
	assign bram_c_enb = (ctrl_state == ST_WRITE);
	assign bram_c_web = (ctrl_state == ST_WRITE) ? 4'hF : 4'h0;
	assign bram_c_dinb = c_word_sel_nxt;

	assign done = done_reg;

	// User logic ends

	endmodule
