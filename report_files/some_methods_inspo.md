## Methods

### General Algorithm

Thhis work introduces a new scheme for the parallelization of the nERF algorithm when applied to polymers. Previous works have explored the division of the polymer in different fragments, the iterative folding for each fragment in parallel, and the concatenation of different fragments. 

Many polymers such as large biomolecules can be divided into a backbone and a sed of ramifications, usually referred to as sidechains. This work takes a parallel scheme similar to the previously described one, but takes it to the extreme by considering only the backbone of the polymer when dividing it in fragments. Since the backbone is usually composed by repeated structures (aminoacid backbone in proteins, pentose and phospate group in nucleic acids, etc), it is natural to consider the minimal repeated unit as a fragment. Therefore, the procedure can be separated into 3 phases: 

1. Composition of the minimal repeated structure **in parallel**
2. Concatenation of the backbone
3. Composition of the ramifications **in parallel**


### Protein Specifics

Since the oxygen atom in the carbonyl group in protein backbones is only linked to the carbon atom, we don't include it in our backbone calculations, but incorporate it as a sidechain addition to the main backbone formed by the N-CA-C atoms of each aminoacid.


#### Implementation details

Special effort is put in generalizing every possible function for an arbitrary number of cases, so that a unique call to a function can do the required calculations for as many cases as possible, thus achieving a near-perfect usage of the processor parallel instructions (CPU-native vector instructions such as SIMD and AVX or GPU massively parallel architecture).