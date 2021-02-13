# mp_nerf
Massively Parallel Natural Extension of Reference Frame


## Installation:

Just clone the repo

You'll need:
* torch > 1.6
* numpy
* einops
* joblib
* sidechainnet: https://github.com/jonathanking/sidechainnet#installation
* manually install `ProDY`, `py3Dmol`, `snakeviz`:
	* `pip install proDy`
	* `pip install py3Dmol`
	* `pip install snakeviz`
	* any other package: `pip install package_name`


* matplotlib (to do diagnostic plots)

## Results: 
* On a 330 AA protein:
<center><img src="experiments/profiler_capture_330.png"></center>

Considerations:
* Only CPU execution (i'll run gpu tests later today)
* actual algorithm is about 1/3 of time: sum(mp_nerf_torch, norm, matmul, ..)
* about 1/2 of time is spent in memory-access patterns, so ideally 2x from here would be possible by optimizing it
* total profiler time should be multiplied by 0.63 to see real time (see execution above without profiler). Profiling slows down the code.