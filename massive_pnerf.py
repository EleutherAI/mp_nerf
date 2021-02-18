import time
import numpy as np
# diff ml
import torch
from einops import repeat
# mine
from data_handler import * 


def get_axis_matrix(a, b, c, norm=True):
    """ Gets an orthonomal basis as a matrix of [e1, e2, e3]. 
        Useful for constructing rotation matrices between planes
        according to the first answer here:
        https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
        Inputs:
        * a: (batch, 3) or (3, ). point(s) of the plane
        * b: (batch, 3) or (3, ). point(s) of the plane
        * c: (batch, 3) or (3, ). point(s) of the plane
        Outputs: orthonomal basis as a matrix of [e1, e2, e3]. calculated as: 
            * e1_ = (c-b)
            * e2_proto = (b-a)
            * e3_ = e1_ ^ e2_proto
            * e2_ = e3_ ^ e1_
            * basis = normalize_by_vectors( [e1_, e2_, e3_] )
        Note: Could be done more by Grahm-Schmidt and extend to N-dimensions
              but this is faster and more intuitive for 3D.
    """
    v1_ = c - b 
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis    = torch.stack([v1_, v2_ready, v3_], dim=-2)
    # normalize if needed
    if norm:
        return basis / torch.norm(basis, dim=-1, keepdim=True) 
    return basis



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
        * seqs: iterable (string, list...) of aas (1 letter corde)
        * cloud_mask: (L, 14) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, 14, L) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom

        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    length = len(seq)
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device).float() # .double()

    # do first AA
    coords[0, 1] = coords[0, 0] + torch.tensor([1, 0, 0], device=device).float() * BB_BUILD_INFO["BONDLENS"]["n-ca"] 
    coords[0, 2] = coords[0, 1] + torch.tensor([torch.cos(np.pi - angles_mask[0, 0, 2]),
                                                torch.sin(np.pi - angles_mask[0, 0, 2]),
                                                0.], device=device).float() * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    
    # starting positions (in the x,y plane) and normal vector [0,0,1]
    init_a = repeat(torch.tensor([1., 0., 0.]).float(), 'd -> l d', l=length).to(device)
    init_b = repeat(torch.tensor([1., 1., 0.]).float(), 'd -> l d', l=length).to(device)
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

    # iteratively join fragments and build the chain backbone
    for i in range(1, length):        
        # move coords and add offset from previous aa
        coords[i, :4] = torch.matmul(coords[i, :4], rotations[i-1]) + coords[i-1, 3].unsqueeze(0)
        # rotate rotation matrix according to previous rotation
        if i < length-1:
            rotations[i] = torch.matmul(rotations[i], rotations[i-1])
        

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



