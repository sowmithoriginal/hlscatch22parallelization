# Catch 22 HLS

This github is a test bed to test teh Catch22 feature extractors.

## Installation 

```
git clone https://github.com/sowmithoriginal/hlscatch22parallelization.git
cd hlscatch22parallelization
```

## Hardware build

```
make build TARGET=hw PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm
make host
./host build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin
```

## Hardware emulation

```
make run TARGET=hw_emu PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm
```
