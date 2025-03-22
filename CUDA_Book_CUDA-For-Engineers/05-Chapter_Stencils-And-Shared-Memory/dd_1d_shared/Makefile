NVCC = /usr/local/cuda-12.8/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -Wno-deprecated-gpu-targets


all: main.exe 

main.exe: main.o kernel.o
	$(NVCC) $^ -o $@ -Wno-deprecated-gpu-targets

main.o: main.cpp kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

kernel.o : kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 