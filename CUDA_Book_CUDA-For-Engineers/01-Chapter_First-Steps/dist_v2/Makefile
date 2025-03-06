NVCC = /usr/local/cuda-12.8/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -Wno-deprecated-gpu-targets
main.exe: main.o aux_functions.o
	$(NVCC) $^ -o $@ -Wno-deprecated-gpu-targets

main.o: main.cpp aux_functions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

aux_functions.o : aux_functions.cpp aux_functions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 