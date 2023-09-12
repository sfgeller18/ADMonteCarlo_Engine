#ifndef MPFR_HELPERS_H
#define MPFR_HELPERS_H

std::string mpfrToString(const mpfr_t& value) {
    char* decimalStr = nullptr;
    mpfr_exp_t exponent;
    decimalStr = mpfr_get_str(NULL, &exponent, 10, 0, value, MPFR_RNDN);

    if (decimalStr) {
        std::string result(decimalStr);
        mpfr_free_str(decimalStr);
        return result;
    } else {
        return "Conversion Error";
    }
}

#endif