// Accuracy of the core's portable exp / log / sincos against the C library.
#include <cmath>
#include <cstdio>

#include "fg3d_backend_cpu.h"

int main()
{
    double ee = 0, el = 0, es = 0;
    for (int i = 0; i <= 200000; ++i) {
        const float x = -30.0f + 60.0f * i / 200000.0f;      // exp
        const double r = std::exp((double)x);
        ee = std::fmax(ee, std::fabs(fg3d_exp(x) - r) / r);
        const float y = std::pow(10.0f, -6.0f + 12.0f * i / 200000.0f);  // log
        el = std::fmax(el, std::fabs(fg3d_log(y) - std::log((double)y)));
        const float a = -20.0f + 40.0f * i / 200000.0f;     // sin/cos
        float s, c;
        fg3d_sincos(a, &s, &c);
        es = std::fmax(es, std::fmax(std::fabs(s - std::sin((double)a)), std::fabs(c - std::cos((double)a))));
    }
    std::printf("exp max rel err %.2e\nlog max abs err %.2e\nsincos max abs err %.2e\n", ee, el, es);
    const bool ok = ee < 1e-6 && el < 2e-6 && es < 2e-6;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
