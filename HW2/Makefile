all: pi.out
.PHONY: all clean

pi.out: pi.o
	clang++ -o pi.out pi.o

pi.o: pi.cpp
	clang++ -O3 -Wall -c pi.cpp

clean:
	rm *.out *.o
