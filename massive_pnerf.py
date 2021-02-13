import time
import numpy as np
# diff ml
import torch
from einops import repeat
# mine
from data_handler import * 


############################
######## FUNCTIONS #########
############################


def dihedral_torch(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


def mp_nerf_torch(a, b, c, l, theta, chi):
    """ Custom Natural extension of Reference Frame. 
        Inputs:
        * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * c: (batch, 3) or (3,). point(s) of the plane, connected to d
        * theta: (batch,) or (float).  angle(s) between b-c-d
        * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    # safety check
    if not ( (-np.pi <= theta) * (theta <= np.pi) ).all().item():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    # calc vecs
    ba = b-a
    cb = c-b
    # calc rotation matrix. based on plane normals and normalized
    n_plane  = torch.cross(ba, cb, dim=-1)
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate   = torch.stack([cb, n_plane_, n_plane], dim=-1)
    rotate  /= torch.norm(rotate, dim=-2, keepdim=True)
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack([-torch.cos(theta),
                     torch.sin(theta) * torch.cos(chi),
                     torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()


def proto_fold(seq, cloud_mask, point_ref_mask, angles_mask, bond_mask,
               device=torch.device("cpu")):
    """ Calcs coords of a protein given it's
        sequence and internal angles.
        Inputs: 
        * seq: string of aas (1 letter corde)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        Output: (L, 14, 3): coordinates. 
        NOTE: convert BB_BUILD_INFO items to tensors in advance 
    """
    length = len(seq)
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device)

    # do first AA
    c_vec = np.pi - angles_mask[0, 0, 2]
    # coords[0, 0, -1] += 0.1
    coords[0, 1] = coords[0, 0] + torch.tensor([1, 0, 0], device=device).float() * BB_BUILD_INFO["BONDLENS"]["n-ca"] 
    coords[0, 2] = coords[0, 1] + torch.tensor([torch.cos(c_vec),
                                                torch.sin(c_vec),
                                                0.], device=device).float() * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    
    # starting positions (in the x,y plane) and normal vector [0,0,1]
    init_a = repeat(torch.tensor([1., 0., 0.]), 'd -> l d', l=length).to(device)
    init_b = repeat(torch.tensor([0., 1., 0.]), 'd -> l d', l=length).to(device)
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
                                   bond_mask[:, 0], thetas, dihedrals)[:]
    # if True:
    #     return coords, cloud_mask

    #########
    # sequential pass to join fragments
    #########
    # part of rotation matrix corresponding to origin - 3 orthogonals
    v1_o  = coords[0, 0] - init_b[0] # we know the norm is 1. # v1_o /= torch.norm(v1_o)
    v2_o  = (init_b[0] - init_a[0]) / np.sqrt(2) # we know the norm # / torch.norm(v2_o) 
    v3_o  = torch.cross(v1_o, v2_o, dim=-1)
    v2_o_ready = torch.cross(v3_o, v1_o, dim=-1)
    mat_origin = torch.stack([v1_o, v2_o_ready, v3_o], dim=-1) 
    # no norm here since vectors are norm=1

    for i in range(1, length):
        # get offset from previous n position
        offset = coords[i-1, 3].unsqueeze(0)
        # part of rotation matrix corresponding to destin - 3 orthogonals
        # we know the norm are the bond lengths
        v1_d  = coords[i-1, 3] - coords[i-1, 2]
        v2_d  = coords[i-1, 2] - coords[i-1, 1]
        v3_d  = torch.cross(v1_d, v2_d, dim=-1)
        v2_d_ready  = torch.cross(v3_d, v1_d, dim=-1)
        mat_destin  = torch.stack([v1_d, v2_d_ready, v3_d], dim=-1)
        mat_destin /= torch.norm(mat_destin, dim=-2, keepdim=True)
        # get rotation matrix
        # rotate  = torch.matmul(mat_destin, mat_origin.t()).t()
        rotate  = torch.matmul(mat_origin, mat_destin.t())
        rotate /= torch.norm(rotate, dim=-1, keepdim=True)
        # move coords
        coords[i, :4] = torch.matmul(coords[i, :4], rotate) + offset

    #########
    # parallel sidechain - do the oxygen, c-beta and side chain
    #########
    for i in range(3,14):
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]
        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # for 1st residue, use position of the second residue's N
            first_next_n     = coords[1, :1] # 1, 3
            # the c requested is from the previous residue
            main_c_prev_idxs = coords[level_mask, idx_a][:-1] # (L-1), 3
            # concat
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[level_mask, idx_a]

        coords[level_mask, i] = mp_nerf_torch(coords_a,
                                                coords[level_mask, idx_b],
                                                coords[level_mask, idx_c],
                                                bond_mask[level_mask, i], thetas, dihedrals)
    
    return coords, cloud_mask


if __name__ == "__main__":
    tests = {"100_A" : ''.join( ["A"]*100  ), 
             "500_A" : ''.join( ["A"]*500  ), 
             "1000_A": ''.join( ["A"]*1000 )
            }
    # test
    for k,v in tests.items():
        tac = time.time()
        coords, mask = proto_fold( v, angles=torch.ones(len(v), 12).float() )
        tic = time.time()
        print("Time for {0} is {1}".format(k, tic-tac))



