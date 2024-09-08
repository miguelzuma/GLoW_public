#############################################################

CC = gcc
CFLAGS = -fPIC -Ofast
LDFLAGS = -lm

INC_DIR = .
LIB_DIR = .

#############################################################

all: test_complex

test_%: test_%.c
	$(CC) -c $(CFLAGS) $< -o $@.o -I$(INC_DIR)
	$(CC) $@.o -L$(LIB_DIR) $(LDFLAGS) -o $@
	rm test_*.o

#############################################################

.PHONY: clean
clean:
	rm -f *.o *.a
	ls test_* | grep -v *.c | xargs rm -f
