
all: dot0 dot3

dot0: dot.c
	gcc -Wall -O0 dot.c -o dot0

dot3: dot.c
	gcc -Wall -O3 dot.c -o dot3
