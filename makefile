#subfolders src tests
.PHONY: install
install:
	make -C ./src install

all:
	make -C ./src all

.PHONY: clean
clean:
	make -C ./src clean
	make -C ./tests clean

.PHONY: test
test:
	make -C ./tests test
