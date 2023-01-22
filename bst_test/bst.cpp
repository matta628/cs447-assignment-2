/*
 * This reads in a binary file of doubles, and inserts into a BST.
 */

#include <random>
#include <cstdlib>
#include <cstdio>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>

class Node {
    public:
        Node(double v) : value{v}, left_child{nullptr}, right_child{nullptr} {}
        static void insert(Node **pp, Node *n);
        static void verify(Node *n);
    private:
        const double value;
        Node *left_child, *right_child;
};

void
Node::insert(Node **pp, Node *n) {
    if (*pp == nullptr) {
        *pp = n;
    } else {
        Node *p = *pp;
        // May have dupes, in theory.
        if (n->value <= p->value) {
            insert(&p->left_child, n);
        } else {
            insert(&p->right_child, n);
        }
    }
    return;
}

void
Node::verify(Node *n) {
    if (n->left_child) {
        assert(n->left_child->value <= n->value);
        verify(n->left_child);
    }
    // std::cerr << std::fixed << std::setprecision(0) << std::setw(7) << n->value << std::endl;
    if (n->right_child) {
        assert(n->right_child->value > n->value);
        verify(n->right_child);
    }
    return;
}

int
main(int argc, char **argv) {

    int rv;

    void set_commas();
    set_commas();

    if (argc != 2) {
        std::fprintf(stderr, "Usage: make <how_many_doubles> <output_file>\n");
        std::exit(1);
    }

    // Many temporary variables here. Hide them in a block.  The only thing we want out
    // of all this is the number of doubles and the pointer to the first.
    double *array = nullptr;
    std::size_t n_doubles = 0;
    {
        int fd = open(argv[1], O_RDONLY);
        if (fd < 0) {
            int en = errno;
            std::fprintf(stderr, "Couldn't open %s: %s\n", argv[1], strerror(en));
            exit(2);
        }

        struct stat sb;
        int rv = fstat(fd, &sb); assert(rv == 0);
        // std::cout << sb.st_size << std::endl;
        // Make sure that the size is a multiple of the size of a double.
        assert(sb.st_size%sizeof(double) == 0);

        void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
        if (vp == MAP_FAILED) {
            int en = errno;
            fprintf(stderr, "mmap() failed: %s\n", strerror(en));
            exit(3);
        }

        rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);

        rv = close(fd); assert(rv == 0);

        array = (double *) vp;
        n_doubles = sb.st_size/sizeof(double);
    }

    // Preallocate memory for nodes.
    Node *nodes = (Node *) ::operator new(n_doubles*sizeof(Node));

    Node *root = nullptr;

    // Main loop.
    for (std::size_t i = 0; i < n_doubles; i++) {
        // Use placement new.
        Node *n = new (nodes + i) Node{array[i]};
        Node::insert(&root, n);
    }

    // For this simple struct, it's not really necessary to call the dtor, but
    // just some defensive programming.
    for (std::size_t i = 0; i < n_doubles; i++) {
        nodes[i].~Node();
    }

    ::operator delete(nodes);

    struct rusage ru;
    rv = getrusage(RUSAGE_SELF, &ru); assert(rv == 0);
    auto cv = [](const timeval &tv) {
        return double(tv.tv_sec) + double(tv.tv_usec)/1000000;
    };
    std::cerr << "Resource Usage:\n";
    std::cerr << "    User CPU Time: " << cv(ru.ru_utime) << '\n';
    std::cerr << "    Sys CPU Time: " << cv(ru.ru_stime) << '\n'; 
    std::cerr << "    Max Resident: " << ru.ru_maxrss << '\n';
    std::cerr << "    Page Faults: " << ru.ru_majflt << '\n';

    /*
    std::cerr << "Verifying..." << std::endl;
    Node::verify(root);
    */
}
