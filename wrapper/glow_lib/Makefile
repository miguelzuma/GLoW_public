objects = lenses_lib.o \
		  roots_lib.o \
		  single_contour_lib.o \
		  analytic_SIS_lib.o \
		  single_integral_lib.o \
		  area_lib.o \
		  special_lib.o \
		  fourier_lib.o \
		  ode_tools.o \
		  contour_lib.o \
		  common.o

test_cases = test_lenses \
			 test_roots \
			 test_single_contour \
			 test_single_integral \
			 test_area \
			 test_special \
		  	 test_fourier \
			 test_contour

#############################################################

INC_DIR = include
SRC_DIR = source
LIB_DIR = lib
TEST_DIR = tests

CC = gcc
CFLAGS = -fPIC -fopenmp -Ofast
LDFLAGS = -fopenmp -lglow -lgsl -lgslcblas -lm

#CFLAGS = -fPIC -g -Wall
#LDFLAGS = -lglow -lgsl -lgslcblas -lm

#############################################################

FFT = pocketfft
LIB = $(LIB_DIR)/libglow.a
OBJECTS = $(addprefix $(LIB_DIR)/, $(objects))
TESTS = $(addprefix $(TEST_DIR)/, $(test_cases))

#############################################################

default: $(LIB)
all: $(TESTS)

$(LIB_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/%.h $(INC_DIR)/common.h | $(LIB_DIR)
	$(CC) -c $(CFLAGS) $< -o $@ -I$(INC_DIR) -I$(FFT)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)
	mkdir -p external

$(LIB_DIR)/$(FFT).o: $(FFT)/$(FFT).c $(FFT)/$(FFT).h
	$(CC) -c $(CFLAGS) $< -o $@ -I$(FFT)

$(LIB): $(OBJECTS) $(LIB_DIR)/$(FFT).o
	ar rcs $(LIB) $^

$(TEST_DIR)/test_%: $(TEST_DIR)/test_%.c $(LIB)
	$(CC) -c $(CFLAGS) $< -o $@.o -I$(INC_DIR)
	$(CC) $@.o -L$(LIB_DIR) $(LDFLAGS) -o $@
	rm $(TEST_DIR)/*.o

test: $(TESTS)

#############################################################

.PHONY: valgrind
valgrind: $(TEST_DIR)/test_contour
	valgrind --leak-check=yes --show-reachable=yes --track-origins=yes $(TEST_DIR)/test_fourier > /dev/null

.PHONY: clean
clean:
	rm -f *.o *.a $(LIB_DIR)/*.o $(LIB_DIR)/*.a external/*_lens.dat
	cd $(TEST_DIR); rm -f $(test_cases); cd ..
