NVCC = /usr/local/cuda-12.8/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -Wno-deprecated-gpu-targets
main.exe: main.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@
