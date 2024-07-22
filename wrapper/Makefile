LIB_DIR = glow_lib
LIB = $(LIB_DIR)/lib/libglow.a

default: setup.py

setup.py: $(LIB)
	python3 setup.py build_ext --inplace -j 4 && rm -Rf build

$(LIB):
	make -C $(LIB_DIR)

.PHONY: clean
clean:
	make -C $(LIB_DIR) clean
	rm -f src/*.c
	rm -f src/*.html
	rm -rf __pycache__
	rm -f *.so
