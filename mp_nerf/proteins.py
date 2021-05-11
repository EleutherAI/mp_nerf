# science
import numpy as np 
# diff / ml
import torch
from einops import repeat
# module
from massive_pnerf import *


def protein_fold(seq, cloud_mask, point_ref_mask, angles_mask, bond_mask,
               device=torch.device("cpu")):
    """ Calcs coords of a protein given it's
        sequence and internal angles.
        Inputs: 
        * seqs: iterable (string, list...) of aas (1 letter corde)
        * cloud_mask: (L, 14) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, 14, L) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom

        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    # automatic type (float, mixed, double) and size detection
    precise = bond_mask.type()
    length  = cloud_mask.shape[0]
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device).type(precise)

    # do first AA
    coords[0, 1] = coords[0, 0] + torch.tensor([1, 0, 0], device=device).type(precise) * BB_BUILD_INFO["BONDLENS"]["n-ca"] 
    coords[0, 2] = coords[0, 1] + torch.tensor([torch.cos(np.pi - angles_mask[0, 0, 2]),
                                                torch.sin(np.pi - angles_mask[0, 0, 2]),
                                                0.], device=device).type(precise) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    
    # starting positions (in the x,y plane) and normal vector [0,0,1]
    init_a = repeat(torch.tensor([1., 0., 0.]).type(precise), 'd -> l d', l=length).to(device)
    init_b = repeat(torch.tensor([1., 1., 0.]).type(precise), 'd -> l d', l=length).to(device)
    # do N -> CA. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 1]
    coords[1:, 1] = mp_nerf_torch(init_a,
                                   init_b, 
                                   coords[:, 0], 
                                   bond_mask[:, 1], thetas, dihedrals)[1:]
    # do CA -> C. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 2]
    coords[1:, 2] = mp_nerf_torch(init_b,
                                   coords[:, 0],
                                   coords[:, 1],
                                   bond_mask[:, 2], thetas, dihedrals)[1:]
    # do C -> N
    thetas, dihedrals = angles_mask[:, :, 0]
    coords[:, 3] = mp_nerf_torch(coords[:, 0],
                                   coords[:, 1],
                                   coords[:, 2],
                                   bond_mask[:, 0], thetas, dihedrals)

    #########
    # sequential pass to join fragments
    #########
    # part of rotation mat corresponding to origin - 3 orthogonals
    mat_origin  = get_axis_matrix(init_a[0], init_b[0], coords[0, 0], norm=False)
    # part of rotation mat corresponding to destins || a, b, c = CA, C, N+1
    # (L-1) since the first is in the origin already 
    mat_destins = get_axis_matrix(coords[:-1, 1], coords[:-1, 2], coords[:-1, 3])

    # get rotation matrices from origins
    # https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    rotations  = torch.matmul(mat_origin.t(), mat_destins)
    rotations /= torch.norm(rotations, dim=-1, keepdim=True)

    # do rotation concatenation - do for loop in cpu always - faster
    rotations = rotations.cpu() if coords.is_cuda else rotations
    for i in range(1, length-1):
        rotations[i] = torch.matmul(rotations[i], rotations[i-1])
    rotations = rotations.to(device) if coords.is_cuda else rotations
    # rotate all
    coords[1:, :4] = torch.matmul(coords[1:, :4], rotations)
    # offset each position by cumulative sum at that position
    coords[1:, :4] += torch.cumsum(coords[:-1, 3], dim=0).unsqueeze(-2)


    #########
    # parallel sidechain - do the oxygen, c-beta and side chain
    #########
    for i in range(3,14):
        level_mask = cloud_mask[:, i]
        # thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # for 1st residue, use position of the second residue's N
            first_next_n     = coords[1, :1] # 1, 3
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            main_c_prev_idxs = coords[(level_mask.nonzero().view(-1) - 1), idx_a][1:] # (L-1), 3
            # concat coords
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[level_mask, idx_a]

        coords[level_mask, i] = mp_nerf_torch(coords_a, 
                                              coords[level_mask, idx_b],
                                              coords[level_mask, idx_c],
                                              bond_mask[level_mask, i], *angles_mask[:, level_mask, i])
    
    return coords, cloud_mask