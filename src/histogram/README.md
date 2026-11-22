# histogram

Counts how often each value occurs in a 1D input of bytes (values 0–255). The output is 256 bins of `uint32_t`, where `bins[v]` is the number of input elements equal to `v`.

Many work-items can hit the same bin at the same time, so every increment has to be an atomic add.
