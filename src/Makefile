CXX        = nvcc -g -G -arch=compute_35 -Wno-deprecated-gpu-targets

SRC1 = main.cu
SOURCES_CPP = main.cu 
HELPERS     = helpers.cu.h cuda-kernels.cu.h cpu-kernels.cu.h
EXECUTABLE1 = bfast-mon

default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE1) $(EXECUTABLE2)

$(EXECUTABLE1): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)


run: $(EXECUTABLE1) 
	./$(EXECUTABLE1)

clean:
	rm -f $(EXECUTABLE1)

