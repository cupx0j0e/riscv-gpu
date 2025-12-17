IVERILOG ?= iverilog
VVP      ?= vvp

RTL_DIR  := ../rtl
SIM      := sim.vvp

SRC      := $(RTL_DIR)/systolic_array_tb.v \
            $(RTL_DIR)/systolic_array.v    \
            $(RTL_DIR)/mac_pe.v

.PHONY: 2x2 run clean

2x2: $(SIM)
	$(VVP) $(SIM)

run: 2x2

$(SIM): $(SRC)
	$(IVERILOG) -o $(SIM) $(SRC)

clean:
	rm -f $(SIM) wave.vcd sim
