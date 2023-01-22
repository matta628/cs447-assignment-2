/*
 * Use this to create a binary file of doubles.
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

int
main(int argc, char **argv) {

    void set_commas();
    set_commas();

    if (argc != 3) {
        std::fprintf(stderr, "Usage: make <how_many_doubles> <output_file>\n");
        std::exit(1);
    }

    std::default_random_engine eng;
    std::uniform_real_distribution<double> dist(0, 1000000);

    std::size_t n;
    {
        std::istringstream ss(argv[1]);
        // Need to input to signed number so we can detect negative numbers.
        // See: https://stackoverflow.com/questions/21092736/how-to-safely-read-an-unsigned-int-from-a-stream
        long long i;
        ss >> i;
        if (ss.fail() || i <= 0) {
            fprintf(stderr, "%s is not a positive integer.\n", ss.str().c_str());
            exit(1);
        }
        assert(i > 0);
        n = i;
    }

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        int en = errno;
        std::fprintf(stderr, "Couldn't open %s: %s\n", argv[1], strerror(en));
        exit(2);
    }

    for (std::size_t i = 0; i < n; i++) {
        double x = dist(eng);
        out.write(reinterpret_cast<char *>(&x), sizeof(x));
        // std::cout << x << std::endl;
    }
}
