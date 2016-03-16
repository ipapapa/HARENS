#subfolders src tests
.PHONY: install
install:
	make -C ./src install

.PHONY: all
all:
	make -C ./src all

.PHONY: lib
lib: 
	make -C ./src lib

.PHONY: clean
clean:
	make -C ./src clean
	make -C ./tests clean
	rm -r lib.d

.PHONY: test
test:
	make -C ./tests test
