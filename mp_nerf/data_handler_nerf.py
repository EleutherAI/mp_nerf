# Author: Eric Alcaide

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np 
from einops import repeat, rearrange
# module
from kb_proteins import SC_BUILD_INFO

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
    if int(torch.__version__.split(".")[1]) >= 7:
        V, S, W = torch.linalg.svd(C.detach()) 
    else: 
        V, S, W = torch.svd(C.detach())
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W.t())
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_


def rmsd_torch(X, Y):
    """ Assumes x,y are both (batch, d, n) - usually (batch, 3, N). """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )


#################
##### DOERS #####
#################

def make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(14)
    if aa != "_":
        n_atoms = 4+len( SC_BUILD_INFO[aa]["atom-names"] )
        mask[:n_atoms] = 1
    return mask

def make_bond_mask(aa):
    """ Gives the length of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    mask[0] = BB_BUILD_INFO["BONDLENS"]['c-n']
    mask[1] = BB_BUILD_INFO["BONDLENS"]['n-ca']
    mask[2] = BB_BUILD_INFO["BONDLENS"]['ca-c']
    mask[3] = BB_BUILD_INFO["BONDLENS"]['c-o']
    # sidechain - except padding token 
    if aa in SC_BUILD_INFO.keys():
        for i,bond in enumerate(SC_BUILD_INFO[aa]['bonds-vals']):
            mask[4+i] = bond
    return mask

def make_theta_mask(aa):
    """ Gives the theta of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    #
    # sidechain
    for i,theta in enumerate(SC_BUILD_INFO[aa]['angles-vals']):
        mask[4+i] = theta
    return mask

def make_torsion_mask(aa):
    """ Gives the dihedral of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    #
    # sidechain
    for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-vals']):
        if torsion == 'p':
            mask[4+i] = np.nan 
        elif torsion == "i":
            # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L372
            mask[4+i] =  999  # anotate to change later # mask[4+i-1] - np.pi
        else:
            mask[4+i] = torsion
    return mask

def make_idx_mask(aa):
    """ Gives the idxs of the 3 previous points. """
    mask = np.zeros((11, 3))
    # backbone
    mask[0, :] = np.arange(3) 
    # sidechain
    mapper = {"N": 0, "CA": 1, "C":2,  "CB": 4}
    for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-names']):
        # get all the atoms forming the dihedral
        torsions = [x.rstrip(" ") for x in torsion.split("-")]
        # for each atom
        for n, torsion in enumerate(torsions[:-1]):
            # get the index of the atom in the coords array
            loc = mapper[torsion] if torsion in mapper.keys() else 4 + SC_BUILD_INFO[aa]['atom-names'].index(torsion)
            # set position to index
            mask[i+1][n] = loc
    return mask


###################
##### GETTERS #####
###################
SUPREME_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                    "bond_mask": make_bond_mask(k),
                    "theta_mask": make_theta_mask(k),
                    "torsion_mask": make_torsion_mask(k),
                    "idx_mask": make_idx_mask(k),
                    } 
                for k in "ARNDCQEGHILKMFPSTWYV_"}

# @jit()
def scn_cloud_mask(seq, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * seqs: (length) iterable of 1-letter aa codes of a protein
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (length, 14) boolean mask 
    """ 
    if coords is not None:
        start = (( rearrange(coords, 'b (l c) d -> b l c d', c=14) != 0 ).sum(dim=-1) != 0).float()
        # if a point is 0, the following are 0s as well
        for b in range(start.shape[0]):
            for pos in range(start.shape[1]):
                for chain in range(start.shape[2]):
                    if start[b, pos, chain].item() == 0.:
                        start[b, pos, chain:] *= 0.
        return start
    return torch.tensor([SUPREME_INFO[aa]['cloud_mask'] for aa in seq])


def scn_bond_mask(seq):
    """ Inputs: 
        * seqs: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 14) maps point to bond length
    """ 
    return torch.tensor([SUPREME_INFO[aa]['bond_mask'] for aa in seq])


def scn_angle_mask(seq, angles):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        * angles: (length, 12). [phi, psi, omega, b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca), 6_scn_torsions]
        Outputs: (L, 14) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    device, precise = angles.device, angles.type()
    angles = angles
    # get masks
    theta_mask   = torch.tensor([SUPREME_INFO[aa]['theta_mask'] for aa in seq]).type(precise)
    torsion_mask = torch.tensor([SUPREME_INFO[aa]['torsion_mask'] for aa in seq]).type(precise)
    # fill masks with angle values
    theta_mask[:, 0] = angles[:, 4] # ca_c_n
    theta_mask[1:, 1] = angles[:-1, 5] # c_n_ca
    theta_mask[:, 2] = angles[:, 3] # n_ca_c
    theta_mask[:, 3] = BB_BUILD_INFO["BONDANGS"]["ca-c-o"]
    # backbone_torsions
    torsion_mask[:, 0] = angles[:, 1] # n determined by psi of previous
    torsion_mask[1:, 1] = angles[:-1, 2] # ca determined by omega of previous
    torsion_mask[:, 2] = angles[:, 0] # c determined by phi
    # O placement - same as in sidechainnet
    # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L313der.py#L313
    torsion_mask[:, 3] = angles[:, 1] - np.pi 
    torsion_mask[-1, 3] += np.pi              
    # add torsions to sidechains
    to_fill = torsion_mask != torsion_mask # "p" fill with passed values
    to_pick = torsion_mask == 999          # "i" infer from previous one
    for i in range(len(seq)):
        # check if any is nan -> fill the holes
        number = to_fill[i].long().sum()
        torsion_mask[i, to_fill[i]] = angles[i, 6:6+number]

        # pick previous value for inferred torsions
        for j, val in enumerate(to_pick[i]):
            if val:
                torsion_mask[i, j] = torsion_mask[i, j-1] - np.pi # pick values from last one.

    return torch.stack([theta_mask, torsion_mask], dim=0).to(device)


def scn_index_mask(seq):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 11, 3) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    idxs = torch.tensor([SUPREME_INFO[aa]['idx_mask'] for aa in seq])
    return rearrange(idxs, 'l s d -> d l s')


def build_scaffolds_from_scn_angles(seq, angles, coords=None, device="auto"):
    """ Builds scaffolds for fast access to data
        Inputs: 
        * seq: string of aas (1 letter code)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        * coords: (L, 3) sidechainnet coords. builds the mask with those instead
                  (better accuracy if modified residues present).
        Outputs:
        * cloud_mask: (L, 14 ) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    # auto infer device and precision
    precise = angles.type()
    if device == "auto":
        device = angles.device

    if coords is not None: 
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else: 
        cloud_mask = scn_cloud_mask(seq)

    cloud_mask = torch.tensor(cloud_mask).bool().to(device)
    
    point_ref_mask = torch.tensor(scn_index_mask(seq)).long().to(device)
     
    angles_mask = torch.tensor(scn_angle_mask(seq, angles)).type(precise).to(device)
     
    bond_mask = torch.tensor(scn_bond_mask(seq)).type(precise).to(device)
    # return all in a dict
    return {"cloud_mask":     cloud_mask, 
            "point_ref_mask": point_ref_mask,
            "angles_mask":    angles_mask,
            "bond_mask":      bond_mask }


#############################
####### ENCODERS ############
#############################


def modify_scaffolds_with_coords(scaffolds, coords):
    """ Gets scaffolds and fills in the right data.
        Inputs: 
        * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
        * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
        Outputs: corrected scaffolds
    """


    # calculate distances and update: 
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1) # N
    scaffolds["bond_mask"][ :, 1] = torch.norm(coords[ :, 1] - coords[:  , 0], dim=-1) # CA
    scaffolds["bond_mask"][ :, 2] = torch.norm(coords[ :, 2] - coords[:  , 1], dim=-1) # C
    # O, CB, side chain
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # get indexes
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][:, :, i-3] # (3, L, 11) -> 3 * (L, 11)
        # correct distances
        scaffolds["bond_mask"][:, i] = torch.norm(coords[:, i] - coords[selector, idx_c], dim=-1)
        # get angles
        scaffolds["angles_mask"][0, :, i] = get_angle(coords[selector, idx_b], 
                                                      coords[selector, idx_c], 
                                                      coords[:, i])
        # handle C-beta, where the C requested is from the previous aa
        if i == 4:
            # for 1st residue, use position of the second residue's N
            first_next_n     = coords[1, :1] # 1, 3
            # the c requested is from the previous residue
            main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]# (L-1), 3
            # concat
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(coords_a,
                                                         coords[selector, idx_b], 
                                                         coords[selector, idx_c], 
                                                         coords[:, i])
    # correct angles and dihedrals for backbone 
    scaffolds["angles_mask"][0, :-1, 0] = get_angle(coords[:-1, 1], coords[:-1, 2], coords[1: , 0]) # ca_c_n
    scaffolds["angles_mask"][0, 1:,  1] = get_angle(coords[:-1, 2], coords[1:,  0], coords[1: , 1]) # c_n_ca
    scaffolds["angles_mask"][0,  :,  2] = get_angle(coords[:,   0], coords[ :,  1], coords[ : , 2]) # n_ca_c
    
    # N determined by previous psi = f(n, ca, c, n+1)
    scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0])
    # CA determined by omega = f(ca, c, n+1, ca+1)
    scaffolds["angles_mask"][1,  1:, 1] = get_dihedral(coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1])
    # C determined by phi = f(c-1, n, ca, c)
    scaffolds["angles_mask"][1,  1:, 2] = get_dihedral(coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2])

    return scaffolds



if __name__ == "__main__":
    print(scn_cloud_mask("AAAA"))
