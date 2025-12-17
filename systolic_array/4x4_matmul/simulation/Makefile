IVERILOG ?= iverilog
VVP      ?= vvp

RTL_DIR  := ../rtl
SIM      := sim.vvp

SRC      := $(RTL_DIR)/systolic_array_4x4_tb.v \
            $(RTL_DIR)/systolic_array_4x4.v    \
            $(RTL_DIR)/pe_4x4.v

.PHONY: 4x4 run clean

4x4: $(SIM)
	$(VVP) $(SIM)

run: 4x4

$(SIM): $(SRC)
	$(IVERILOG) -o $(SIM) $(SRC)

clean:
	rm -f $(SIM) wave.vcd sim
