## Introduction

Molecular modelling often uses two distinct sets of coordinates in order to represent polymers like proteins or nucleic acids: internal coordinates (bond lengths, bond angles and dihedrals) and cartesian coordinates (x,y,z). These 2 coordinates systems allow for a complete representation of the molecule, and have both their strengths and weakneses: cartesian coordinates are easier to work with when dealing with tasks that treat the polymer as a rigid system (mainly rotations, translations or visualizations), whereas internal coordinates are preferred when working with forces and interactions between the atoms which form the polymer.

The translation between these systems, however, can become a bottleneck in many applications. While the conversion from cartesian to internal coordinates (often referred to as *forward translation* in the literature) can be straightforwardly parallelized across all polymer points in modern hardware (since the internal coordinates can be calculated independently for each point), it is not the case in the reverse problem (often referred to as *reverse translation*). The translation from internal coordinates to cartesian ones needs to be carried out sequentially since the position of each atom depends on the position of the previous one, in addition to the bond length, bond angle and dihedral angle.

This dependency across the polymer length imposes a time constraint to the *reverse translation* and, although the calculation can be parallelized across many polymers of similar length, the bottleneck is significant for applications that make intensive use of the forward and reverse translations. Examples of such applications include the training of machine learning models, protein structure refinement in the processing of the data obtained by NMR [ref 1 of sota paper], analysis of protein structure changes and molelucar dynamics simulations, etc.

The standard algorithm used for the *reverse translation* recieves the name of NERF (Natural Extension of Reference Frame). In recent years, with the development of more and better computational tools, some effort has been devoted into alleviating the bottleneck that the reverse translation presents. These works have focused on the usage of high-performance, optimized code, the division of the polymer into different fragments (which are folded independently and later ensembled) [pnerf paper alquraishi] and tree ensembling algorithms [sota paper] among others. 

However, this approaches are often implemented specifically for Graphical Processing Units (GPU) to take advantage of its massively parallel architecture, thus limiting the potential usage due to the expensive and specialized hardware needed. tIn addition to that he translation from internal coordinates to cartesian ones continues to be a bottleneck in many pipelines that use it.

This work introduces a massively-parallel NERF algorithm, which reorders the code to explore parallelization and an optimized usage of instructions when possible. The algorithm presented combines features from previous work, exploits some of its ideas, and builds on top of them to provide an acceleration about 300x-1200x, depending on the length of the polymer being translated. 

The main contribution of this work is a massively-parallel Natural extension of Reference Frame (mp-NeRF) algorithm, which acieves speedups of 2-3 orders of magnitude with respect to the previous state of the art. This algorithm allows for an increased efficiency of pipelines which make heavy use of such conversions, like the training of machine learning models to predict polymer structures or different features. 


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

#### Implementation details

* The calculations for the rotation matrices to join the backbone fragments are decomposed into a rotation from the $N$ to $$N_{+1}$$ and the rotation from the $$N_{0}$$ to the $${N}$$ minimal structure, thus allowing for the parallelization of the rotation matrices in the base position, leaving a cumulative matrix multiplication as the only sequential part of the algorithm that can not be parallelized.
* The translation operation is implemented as a cummulative sum, accelerating its calculation.
* Special effort is put in generalizing every possible function for an arbitrary number of cases, so that a unique call to a function can do the required calculations for as many cases as possible, thus achieving a near-perfect usage of the processor native parallel capabilities (CPU-native vector instructions such as SIMD and AVX or GPU massively parallel architecture).
* Since the oxygen atom in the carbonyl group in protein backbones is only linked to the carbon atom, we don't include it in our backbone calculations, but incorporate it as a sidechain addition to the main backbone formed by the N-CA-C atoms of each aminoacid.
We leave the calculation of all sidechains, including C-beta to the *ramifications* step referenced above.

Experiments were conducted on a MacBook Pro with Intel i5, 4nucleus at 2.5Ghz (laptop) and a DXG-1 with .... specs.

### Experiments
We perform 2 experiments in order to benchmark the speed of our algorithm against the previous SOTA and to check the cummulative error that comes with accumulated transformations from internal to cartesian and the reverse mapping (this is of special importantce in algorithms that may use internal coordinates to work with, but that need the conversion functions before and after since the molecular simulation or base representation is in cartesian coordinates)

TABLE 2: Comparison of execution times between previous sota and our implementation
Here we provide time comparisons of our algorithm (both cpu-laptop, cpu-high end and GPU-high end) with previous state of the art algorithms. Our algorithm achieves 1000x improvements over the rpevious state of the art, while being able to run on CPUs (which makes it broadly applicable and able to exploit parallelism in CPU cluster setups).


FIGURE 2: Same info as in table 2 but more points so that a clear curve can be observed

FIGURE 3: Cummulative error (RMSD, not 3 axes) as a function of encoding-decoding phases. 
We can see instability on the error at the beginning (same in the 3-axis setup) likely caused by the numerical stability at initial conditions (floating point precision issues). After the loss stabilizes, we can see a very small increase with the increase in the number of back and forth conversions. 
    

### Discussion / Design Choices / Future Work / Comments

* We would like to emphasize that some design choices are currently limiting the implementation exposed here in terms of speed, but they might be crucial for the adoption of the methods and the scheme proposed. Some of these design choices that carry inevitable trade-offs are:

    * Programming language (Python vs Compiled): the current implementation is written in Python, a high-level language  widely known among the scientific community, with many scientific software packages implemented in it. Since Python is an interpreted language, and thus slow, we estimate that the current implementation could be accelerated by 2x if a switch to a compiled language is done (ex. C++, Rust, ...). However, this would inevitably lead to a reduction in adoption (since those languages are not as used by the community), increased cost of code maintainability and a reduction on the possible extensions, adaptations, reusability and readability of the current implementation.
    * Differentiability (PyTorch vs NumPy): our implementation is differentiable, thus allowing to train Machine Learning / Deep Learning models with it (ex. RGN-Networks https://github.com/aqlaboratory/rgn), which make heavy use of the conversion from internal representation to cartesian coordinates and have been the primary driver of the algorithmic improvements in the recent years. However, the differentiability of the code makes it inevitably slower because the data structures need to accumulate important information for the gradient calculations, and also because the maintainance of this property prevents us from using more efficient libraries like NumPy, that allow compilation of the code to C (Cython, Numba) to achieve faster runtimes.
    * Precision: we perform all our calculations in standard floating point precision (float32) as it is the standard for many applications that perform the internal-to-cartesian conversion such as Machine Learning pipelines. This can result in errors up to 2 orders of magnitude higher than the previous state of the art (error in the range of 1e-2 A while previous sota was in the 1e-4 order).


However, these features could be adapted for a specific case in which a particular set of properties might be preferred over another (ex. single-thread speed over differentiability, parallelization over single-thread speed, ...). We leave the possible further optimizations or adaptations to very specific scenarios to the community.

- In aggregate, we estimate that a reduction about 3x - 5x of the actual CPU runtimes could be possible by adopting a scheme focused on single-thread speed above everything else. 

## Conclusion

In this work have proposed a new massively parallel scheme for the implementation of the Natural Extension of Reference Frame when applied to polymers and showcased the improvement over previous works when applied to proteins.

The design principles put in practice allow for a frictionless adaptability and usage accross the community for different kinds of polymers such as proteins, nucleic acids, glycosaminoglycans or synthetic materials.

We hope this accelerated implementation can reduce the times for computational simulations, accelerate the training of machine learning models, and open the window to new advances in polymer structural science.







