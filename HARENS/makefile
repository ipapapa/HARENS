#This is a makefile for HARENS
#Please make sure you have installed nvcc in your computer and added it in system path

cp = nvcc
flags = -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /O2 /Zi  /MD
objects := $(patsubst %.cpp ,%.o, $(wildcard *.cpp))

HARENS: $(objects)
	$(cp) -M $(flags) $(wildcard *.cpp) > $(objects)
	$(cp) -o HARENS $(objects)

.PHONY: clean
clean:
        rm *.o temp