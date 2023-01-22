.PHONY: all
all: make bst

.PHONY: make
make:
	g++ -O4 -o make make.cpp set_commas.cpp

.PHONY: bst
bst:
	g++ -O4 -o bst bst.cpp set_commas.cpp

.PHONY: clean
clean:
	rm -f *.o bst make core vgcore.* a.out
