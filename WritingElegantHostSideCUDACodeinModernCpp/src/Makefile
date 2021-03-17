
CXXFLAGS = -O3
CC = nvcc

# Note: This will only work for CUDA 9.0 and later
CXXFLAGS += -Wno-deprecated-gpu-targets --std=c++14
CXXFLAGS += -g
#CXXFLAGS += --compiler-options -flto
CXXFLAGS += -I/usr/local/cuda/samples/common/inc/
CXXFLAGS += -Icuda-api-wrappers/include

all: vectorAdd

archive:
	rm -f lab.tar.gz
	tar cvfz lab.tar.gz *.cu *.cuh *.txt *.sh Makefile check_all do_shift meld_next

clone:
	git clone https://github.com/eyalroz/cuda-api-wrappers.git cuda-api-wrappers

cpp14:
	sed -ri 's/std::unique_ptr<float>\(new float\[numElements\]\)/std::make_unique<float\[\]>\(numElements\)/;' *.cu

cpp11:
	sed -ri 's/std::make_unique<float\[\]>\(numElements\)/std::unique_ptr<float>\(new float\[numElements\]\)/;' *.cu

tags:
	ctags --langmap=c++:+.cu.cuh.hpp  -R ./ 

v01: clean
	cp 01.cu vectorAdd.cu

v02: clean
	cp 02.cu vectorAdd.cu

v03: clean
	cp 03.cu vectorAdd.cu

v04: clean
	cp 04.cu vectorAdd.cu

v05: clean
	cp 05.cu vectorAdd.cu

v06: clean
	cp 06.cu vectorAdd.cu

v07: clean
	cp 07.cu vectorAdd.cu

v08: clean
	cp 08.cu vectorAdd.cu

v09: clean
	cp 09.cu vectorAdd.cu

v10: clean
	cp 10.cu vectorAdd.cu

v11: clean
	cp 11.cu vectorAdd.cu

v12: clean
	cp 12.cu vectorAdd.cu

v13: clean
	cp 13.cu vectorAdd.cu

v14: clean
	cp 14.cu vectorAdd.cu

v15: clean
	cp 15.cu vectorAdd.cu

v16: clean
	cp 16.cu vectorAdd.cu

v17: clean
	cp 17.cu vectorAdd.cu

v18: clean
	cp 18.cu vectorAdd.cu

v19: clean
	cp 19.cu vectorAdd.cu

v20: clean
	cp 20.cu vectorAdd.cu

v21: clean
	cp 21.cu vectorAdd.cu

vectorAdd: vectorAdd.o not_interesting.cuh
	$(CC) $(CXXFLAGS) -o vectorAdd vectorAdd.o

clean:
	rm -f *.o *.s *.ptx vectorAdd

%.o: %.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

%.o: %.cc
	$(CC) $(CXXFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CXXFLAGS) -o $@ -c $<

