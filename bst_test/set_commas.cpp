#include <locale>
#include <iostream>

namespace {
    class Comma_numpunct : public std::numpunct<char>
    {
      protected:
        virtual char do_thousands_sep() const {
            return ',';
        }

        virtual std::string do_grouping() const {
            return "\03";
        }
    };
    std::locale Comma_locale(std::locale(), new Comma_numpunct());
}

void
set_commas() {
    std::cout.imbue(Comma_locale);
    std::cerr.imbue(Comma_locale);
}
