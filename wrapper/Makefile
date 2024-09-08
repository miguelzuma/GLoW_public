test_cases = test_gsl \
			 test_omp \
			 test_complex

TEST_DIR = tests
TESTS = $(addprefix $(TEST_DIR)/, $(test_cases))

LIB_DIR = glow_lib
LIB = $(LIB_DIR)/lib/libglow.a

## --------------------------------------------------------

default: $(LIB)
	python3 setup.py build_ext -j 4 --inplace && rm -Rf build

$(LIB):
	make -C $(LIB_DIR) -j

.PHONY: clean
clean:
	for t in $(TESTS); do \
		mv $$t/Makefile.bak $$t/Makefile; \
		make -C $$t clean; \
	done
	-mv $(LIB_DIR)/Makefile.bak $(LIB_DIR)/Makefile
	make -C $(LIB_DIR) clean
	-mv setup.py.bak setup.py
	rm -rf build
	rm -rf temp
	rm -f src/*.c
	rm -f src/*.html
	rm -rf __pycache__
	rm -f *.so
