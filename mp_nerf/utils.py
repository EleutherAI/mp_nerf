# Author: Eric Alcaide

import torch
import numpy as np 
from einops import repeat, rearrange

# data utils
def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, verbose=True):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training. 
        Inputs: 
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
        Outputs: (cleaned, without padding)
        (seq_str, int_seq, coords, angles, padding_seq, mask, pid)
    """
    while True:
        for b,batch in enumerate(dataloader_['train']):
            for i in range(batch.int_seqs.shape[0]):
                # strip padding - matching angles to string means
                # only accepting prots with no missing residues (angles would be 0)
                padding_seq = (batch.int_seqs[i] == 20).sum().item()
                padding_angles = (torch.abs(batch.angs[i]).sum(dim=-1) == 0).long().sum().item()

                if padding_seq == padding_angles:
                    # check for appropiate length
                    real_len = batch.int_seqs[i].shape[0] - padding_seq
                    if max_len >= real_len >= min_len:
                        # strip padding tokens
                        seq = ''.join([vocab_.int2char(aa) for aa in batch.int_seqs[i].numpy()])
                        seq = seq[:-padding_seq or None]
                        int_seq = batch.int_seqs[i][:-padding_seq or None]
                        angles  = batch.angs[i][:-padding_seq or None]
                        mask    = batch.msks[i][:-padding_seq or None]
                        coords  = batch.crds[i][:-padding_seq*14 or None]

                        if verbose:
                            print("stopping at sequence of length", real_len)
                        return seq, int_seq, coords, angles, padding_seq, mask, batch.pids[i]
                    else:
                        if verbose:
                            print("found a seq of length:", batch.int_seqs[i].shape,
                                  "but oustide the threshold:", min_len, max_len)
                else:
                    if verbose:
                        print("paddings not matching", padding_seq, padding_angles)
                    pass

    return None
    


######################
## structural utils ##
######################

def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


def get_angle(c1, c2, c3):
    """ Returns the angle in radians.
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # dont use acos since norms involved. 
    # better use atan2 formula: atan2(cross, dot) from here: 
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # add a minus since we want the angle in reversed order - sidechainnet issues
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1), 
                        -(u1*u2).sum(dim=-1) ) 


def kabsch_torch(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (D, N) - usually (3, N)
    """
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t())
    # Optimal rotation matrix via SVD - warning! W must be transposed
    if int(torch.__version__.split(".")[1]) < 8:
        V, S, W = torch.svd(C.detach())
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C.detach()) 
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_


def rmsd_torch(X, Y):
    """ Assumes x,y are both (batch, d, n) - usually (batch, 3, N). """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )



