# clang++ -O3 -mavx2 -mfma -I ./include -shared -std=c++11 `python-config --cflags --ldflags` src/glove_bind.cpp -o libs/glove/_glove.so



CC = clang++
#For older compilers use -O3 or -O2 instead of -Ofast
CFLAGS = -Ofast -march=native -mavx2 -mfma -funroll-loops  -I ./include -shared -std=c++11 `python-config --cflags --ldflags`

all: glove

glove : src/glove_bind.cpp
	$(CC) $(CFLAGS) src/glove_bind.cpp -o libs/glove/_glove.so
