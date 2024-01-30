# ising-CUDA

## File structure
* src/include: Contain the source and header files for all the implementation functions
* main: Contains all the scripts that are used for the proof of correctness and execution time measurements.
* batch: The sbatch scripts that were used to HPC-Aristotelis for the execution of the scripts.
* bin: Contains binaries.

## Scripts:
* v0visual (make v0visual): (Usage ./bin/v0visual n k) Prints the grid evolution stage by stage.
* v(i)test (make v(i)test): (Usage ./bin/v(i)test) Tests the i-th implementation assuming (i-1)-th implementation is correct.
* v(2)time (make v(i)time): (Usage ./bin/v(i)time dataLocation (threadBlockLength)) According to the script parameters stores at the specified the file dataLocation/v(i)time.txt the median time results in the form: n k initialization_time evolution_time
* threadBlockLength is a parameter only for v2time to get different results for different block sizes.
