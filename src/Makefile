CXX        = gcc -O3# -arch=compute_35 -Wno-deprecated-gpu-targets

SRC1 = main.cu
SOURCES_CPP = main.cu
HELPERS     = cpu-kernels.cu.h cuda-kernels.cu.h helpers.cu
EXECUTABLE1 = bfast-monitor

default: compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE1) $(EXECUTABLE2)

$(EXECUTABLE1): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)


run: $(EXECUTABLE1) 
	./$(EXECUTABLE1)

clean:
	rm -f $(EXECUTABLE1)

