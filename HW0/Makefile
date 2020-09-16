all: pi.out
.PHONY: all clean

pi.out: pi.o
	g++ -o pi.out pi.o

pi.o: pi.cpp
	g++ -g -std=c++11 -O0 -Wall -c pi.cpp

clean:
	rm pi.out pi.o
