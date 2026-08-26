
main: main.c float32.o int32.o int8.o
	gcc -Wall -O3 -o main main.c float32.o int32.o int8.o -lm

float32.o: float32.c float32.h
	gcc -Wall -O3 -c float32.c

int32.o: int32.c int32.h
	gcc -Wall -O3 -c int32.c

int8.o: int8.c int8.h
	gcc -Wall -O3 -c int8.c

clean:
	rm -f main float32.o int32.o int8.o
