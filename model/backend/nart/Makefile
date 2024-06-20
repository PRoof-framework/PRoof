#
# Makefile to compile _nart_run pybind11 python module
#

nart_repo_path = ../../../../NART
target = _nart_run$(shell python3-config --extension-suffix)

$(target): _nart_run.o
	c++ -fPIC -shared -o $(target) _nart_run.o -Wl,-rpath,$(nart_repo_path)/build/art $(nart_repo_path)/build/art/libart.so

_nart_run.o:
	c++ -O3 -Wall -shared -c -std=c++11 -fPIC $(shell python3 -m pybind11 --includes) -I$(nart_repo_path)/art/include/ _nart_run.cpp -o _nart_run.o

.PHONY : clean
clean:
	rm -f $(target) _nart_run.o
