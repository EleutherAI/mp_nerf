## Methods

To render nicely go to: [dillinger.io](https://dillinger.io/)

### General Algorithm

This work introduces a new scheme for the parallelization of the nERF algorithm when applied to polymers. Previous works have explored the division of the polymer in different fragments, the iterative folding for each fragment in parallel, and the concatenation of different fragments. 

Many polymers such as large biomolecules can be divided into a backbone and a sed of ramifications, usually referred to as sidechains. This work takes a parallel scheme similar to the previously described one, but takes it to the extreme by considering only the backbone of the polymer when dividing it in fragments. Since the backbone is usually composed by repeated structures (aminoacid backbone in proteins, pentose and phospate group in nucleic acids, etc), it is natural to consider the minimal repeated unit as a fragment. Therefore, the procedure can be separated into 3 phases: 

1. Composition of the minimal repeated structure **in parallel**

For every minimal repeated structure, we initialize 2 points near the origin coordinates, and take the first point of the structure as the origin. From there, we implement sequentially the NERF algorithm for every point in the minimal structure (in parallel for all the minimal structures) until we reach the first point of the next minimal structure (Figure 1 illustrates the process - TODO: Figure with colours decribing the order). 
This step requires $$n * l$$ NERF calls where $$n$$ is the number of points in the minimal repeated structure and $$l$$ is the number of structures across which the calculation can be parallelized.

2. Assembling of backbone fragments

We then join the different fragments by a rototranslation operation. That is, we move the first point of the $$N_{+1}$$ to the last point of the $$N$$ minimal structure, and we rotate the $$N_{+1}$$ structure to match the orientation of the $$N$$ structure. 
The sequential pass requires $$l-1$$ sequential matrix multiplications (where $$l$$ is the number of minimal structures to assemble) and a cumulative sum of length $$l$$

$$Nnew_{i} = \sum_{j=0}^{i} N_{j}$$ .

3. Composition of the ramifications **in parallel**

After the backbone assembling, we perform the calculation of the ramifications, in parallel for all of them. This requires a maximum of $$n * l$$ NERF calls, where $$n$$ is the maximum number of points any possible ramification, and $$l$$ is the number of ramifications. Note that this is an upper bound, since not all ramifications will have the same number of amount, thus many calculations will not be needed in the end.

FIGURE 1: Description of the algorithm (coloured balls).

#### Protein Specifics

Since the oxygen atom in the carbonyl group in protein backbones is only linked to the carbon atom, we don't include it in our backbone calculations, but incorporate it as a sidechain addition to the main backbone formed by the N-CA-C atoms of each aminoacid.
We leave the calculation of all sidechains, including C-beta to the *ramifications* step referenced above.

### Experiments
We perform 2 experiments in order to benchmark the speed of our algorithm against the previous SOTA and to check the cummulative error that comes with accumulated transformations from internal to cartesian and the reverse mapping (this is of special importantce in algorithms that may use internal coordinates to work with, but that need the conversion functions before and after since the molecular simulation or base representation is in cartesian coordinates)

TABLE 2: Comparison of execution times between previous sota and our implementation

FIGURE 2: Same info as in table 2 but more points so that a clear curve can be observed

FIGURE 3: Cummulative error (RMSD, not 3 axes) as a function of encoding-decoding phases. 


### Implementation details

Special effort is put in generalizing every possible function for an arbitrary number of cases, so that a unique call to a function can do the required calculations for as many cases as possible, thus achieving a near-perfect usage of the processor native parallel capabilities (CPU-native vector instructions such as SIMD and AVX or GPU massively parallel architecture).

* Concatenation of the backbone:
    * The calculations for the rotation matrices to join the backbone fragments are decomposed into a rotation from the $N$ to $$N_{+1}$$ and the rotation from the $$N_{0}$$ to the $${N}$$ minimal structure, thus allowing for the parallelization of the rotation matrices in the base position, leaving a cumulative matrix multiplication as the only sequential part of the algorithm that can not be parallelized.
    * The translation operation is implemented as a cummulative sum, accelerating its calculation.

### Discussion / Design Choices / Future Work / Comments

* We would like to emphasize that some design choices are currently limiting the implementation exposed here in terms of speed, but they might be crucial for the adoption of the methods and the scheme proposed. Some of these design choices that carry inevitable trade-offs are:

    * Programming language (Python vs Compiled): the current implementation is written in Python, a high-level language  widely known among the scientific community, with many scientific software packages implemented in it. Since Python is an interpreted language, and thus slow, we estimate that the current implementation could be accelerated by 2x if a switch to a compiled language is done (ex. C++, Rust, ...). However, this would inevitably lead to a reduction in adoption (since those languages are not as used by the community), increased cost of code maintainability and a reduction on the possible extensions, adaptations, reusability and readability of the current implementation.
    * Differentiability (PyTorch vs NumPy): our implementation is differentiable, thus allowing to train Machine Learning / Deep Learning models with it (ex. RGN-Networks https://github.com/aqlaboratory/rgn), which make heavy use of the conversion from internal representation to cartesian coordinates and have been the primary driver of the algorithmic improvements in the recent years. However, the differentiability of the code makes it inevitably slower because the data structures need to accumulate important information for the gradient calculations, and also because the maintainance of this property prevents us from using more efficient libraries like NumPy, that allow compilation of the code to C (Cython, Numba) to achieve faster runtimes.


However, these features could be adapted for a specific case in which a particular set of properties might be preferred over another (ex. single-thread speed over differentiability, parallelization over single-thread speed, ...). We leave the possible further optimizations or adaptations to very specific scenarios to the community.

- In aggregate, we estimate that a reduction about 3x - 5x of the actual CPU runtimes could be possible by adopting a scheme focused on single-thread speed above everything else. 

## Conclusion

* We have proposed a new parallel scheme for the implementation of the Natural Extension of Reference Frame when applied to polymers and showcased the improvement over previous works when applied to proteins. 

* We hope this accelerated implementation can reduce the times for computational simulations, accelerate ML models' training and open the window to new advances in polymer structural science.







