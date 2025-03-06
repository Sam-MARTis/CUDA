NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -Wno-deprecated-gpu-targets

main.out: kernel.cu 
	$(NVCC) $(NVCC_FLAGS) $^ -o $@