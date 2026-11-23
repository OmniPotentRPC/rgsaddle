The all-features suite no longer stalls in `colour_block_strong_scaling_on_vacancy_cells`.
That test recovers shifted-force Hessians of a 10975-atom vacancy and a 97555-atom vacancy, five times at each thread count.
The test build compiled the neighbour walk unoptimized, so the test stayed in that walk for hours.
Test builds now use opt-level 3, with debug assertions still on.
A 12³ vacancy recovery that took 105s unoptimized takes 13s optimized, and the test fails if that recovery exceeds one minute.
The C ABI smoke tests also find `librgsaddle.so` in the cdylib unit directory, which is where Cargo leaves it when the test profile's opt-level differs from dev.
