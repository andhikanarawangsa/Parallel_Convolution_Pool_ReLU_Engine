# Makefile - Streaming Convolution + Maxpooling + ReLU Accelerator
# Requires: Icarus Verilog (iverilog, vvp). Optional: GTKWave, Python 3.

RTL = rtl/line_buffer.v rtl/window_gen.v rtl/conv_pe.v rtl/conv_array.v \
      rtl/maxpool2x2.v rtl/relu_sat.v rtl/frame_ctrl.v rtl/conv_pool_relu_top.v
TB  = tb/tb_conv_pool_relu_top.v
OUT = sim

.PHONY: all sim sim_shift8 golden wave clean

all: sim

sim:
	mkdir -p $(OUT)
	iverilog -g2005 -Wall -o $(OUT)/sim.vvp $(RTL) $(TB)
	vvp $(OUT)/sim.vvp | tee $(OUT)/sim.log

sim_shift8:
	mkdir -p $(OUT)
	iverilog -g2005 -Wall -Ptb_conv_pool_relu_top.OUT_SHIFT=8 -o $(OUT)/sim_shift8.vvp $(RTL) $(TB)
	vvp $(OUT)/sim_shift8.vvp | tee $(OUT)/sim_shift8.log

golden:
	python3 scripts/golden_model.py | tee $(OUT)/golden.log

wave:
	gtkwave $(OUT)/tb_conv_pool_relu_top.vcd &

clean:
	rm -f $(OUT)/sim.vvp $(OUT)/sim.log $(OUT)/sim_shift8.vvp $(OUT)/sim_shift8.log $(OUT)/golden.log $(OUT)/*.vcd
