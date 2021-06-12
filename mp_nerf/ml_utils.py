# Author: Eric Alcaide

import torch
import numpy as np 
from einops import repeat, rearrange

# module
from mp_nerf.massive_pnerf import *
from mp_nerf.utils import *
from mp_nerf.kb_proteins import *
from mp_nerf.proteins import *


def atom_selector(scn_seq, x, option=None, discard_absent=True): 
    """ Returns a selection of the atoms in a protein. 
        Inputs: 
        * scn_seq: (batch, len) sidechainnet format or list of strings
        * x: (batch, (len * n_aa), dims) sidechainnet format
        * option: one of [torch.tensor, 'backbone-only', 'backbone-with-cbeta',
                  'all', 'backbone-with-oxygen', 'backbone-with-cbeta-and-oxygen']
        * discard_absent: bool. Whether to discard the points for which
                          there are no labels (bad recordings)
    """
    

    # get mask
    present = []
    for i,seq in enumerate(scn_seq): 
        pass_x = x[i] if discard_absent else None
        if pass_x is None and isinstance(seq, torch.Tensor):
            seq = "".join([INDEX2AAS[x] for x in seq.cpu().detach().tolist()])

        present.append( scn_cloud_mask(seq, coords=pass_x) )

    present = torch.stack(present, dim=0).bool()

    
    # atom mask
    if isinstance(option, str):
        atom_mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if "backbone" in option: 
            atom_mask[[0, 2]] = 1

        if option == "backbone": 
            pass
        elif option == 'backbone-with-oxygen':
            atom_mask[3] = 1
        elif option == 'backbone-with-cbeta':
            atom_mask[5] = 1
        elif option == 'backbone-with-cbeta-and-oxygen':
            atom_mask[3] = 1
            atom_mask[5] = 1
        elif option == 'all':
            atom_mask[:] = 1
        else: 
            print("Your string doesn't match any option.")
            
    elif isinstance(option, torch.Tensor):
        atom_mask = option
    else:
        raise ValueError('option needs to be a valid string or a mask tensor of shape (14,) ')
    
    mask = rearrange(present * atom_mask.unsqueeze(0).unsqueeze(0).bool(), 'b l c -> b (l c)')
    return x[mask], mask


def noise_internals(seq, angles=None, coords=None, noise_scale=0.5, theta_scale=0.5, verbose=0):
    """ Noises the internal coordinates -> dihedral and bond angles. 
        Inputs: 
        * seq: string. Sequence in FASTA format
        * angles: (l, 11) sidechainnet angles tensor
        * coords: (l, 14, 13)
        * noise_scale: float. std of noise gaussian.
        * theta_scale: float. multiplier for bond angles
        Outputs: 
        * chain (l, c, d)
        * cloud_mask (l, c)
    """
    assert angles is not None or coords is not None, \
           "You must pass either angles or coordinates"
    # get scaffolds
    if angles is None:
        angles = torch.randn(coords.shape[0], 12).to(coords.device)
        
    scaffolds = build_scaffolds_from_scn_angles(seq, angles.clone())
    
    if coords is not None:
        scaffolds = modify_scaffolds_with_coords(scaffolds, coords)
    
    # noise bond angles and dihedrals (dihedrals of everyone, angles only of BB)
    if noise_scale > 0.:
        if verbose: 
            print("noising", noise_scale)
        # thetas (half of noise of dihedrals. only for BB)
        noised_bb = scaffolds["angles_mask"][0, :, :3].clone()
        noised_bb += theta_scale*noise_scale * torch.randn_like(noised_bb) 
        # get noised values between [-pi, pi]
        off_bounds = (noised_bb > 2*np.pi) + (noised_bb < -2*np.pi)
        if off_bounds.sum().item() > 0: 
            noised_bb[off_bounds] = noised_bb[off_bounds] % (2*np.pi)
            
        upper, lower = noised_bb > np.pi, noised_bb < -np.pi 
        if upper.sum().item() > 0:
            noised_bb[upper] = - ( 2*np.pi - noised_bb[upper] ).clone()
        if lower.sum().item() > 0:
            noised_bb[lower] = 2*np.pi + noised_bb[lower].clone()
        scaffolds["angles_mask"][0, :, :3] = noised_bb

        # dihedrals
        noised_dihedrals = scaffolds["angles_mask"][1].clone()
        noised_dihedrals += noise_scale * torch.randn_like(noised_dihedrals)
        # get noised values between [-pi, pi]
        off_bounds = (noised_dihedrals > 2*np.pi) + (noised_dihedrals < -2*np.pi)
        if off_bounds.sum().item() > 0: 
            noised_dihedrals[off_bounds] = noised_dihedrals[off_bounds] % (2*np.pi)
            
        upper, lower = noised_dihedrals > np.pi, noised_dihedrals < -np.pi 
        if upper.sum().item() > 0:
            noised_dihedrals[upper] = - ( 2*np.pi - noised_dihedrals[upper] ).clone()
        if lower.sum().item() > 0:
            noised_dihedrals[lower] = 2*np.pi + noised_dihedrals[lower].clone()
        scaffolds["angles_mask"][1] = noised_dihedrals
    
    # reconstruct
    return protein_fold(**scaffolds)


def combine_noise(true_coords, seq=None, int_seq=None, angles=None,
                  NOISE_INTERNALS=1e-2, SIDECHAIN_RECONSTRUCT=True):
    """ Combines noises. For internal noise, no points can be missing. 
        Inputs: 
        * true_coords: ((B), N, D)
        * int_seq: (N,) torch long tensor of sidechainnet AA tokens 
        * seq: str of length N. FASTA AAs.
        * angles: (N_aa, D_). optional. used for internal noising
        Outputs: (B, N, D) coords and (B, N) boolean mask
    """
    # get seqs right
    assert int_seq is not None or seq is not None, "Either int_seq or seq must be passed"
    if int_seq is not None and seq is None: 
    	seq = "".join([INDEX2AAS[x] for x in int_seq.cpu().detach().tolist()])
    elif int_seq is None and seq is not None: 
    	int_seq = torch.tensor([AAS2INDEX[x] for x in seq.upper()], device=true_coords.device)

    cloud_mask_flat = (true_coords == 0.).sum(dim=-1) != true_coords.shape[-1]
    naive_cloud_mask = scn_cloud_mask(seq).bool()
    
    if NOISE_INTERNALS: 
        assert cloud_mask_flat.sum().item() == naive_cloud_mask.sum().item(), \
               "atoms missing: {0}".format( naive_cloud_mask.sum().item() - \
                                            cloud_mask_flat.sum().item() )
    # expand to batch dim if needed
    if len(true_coords.shape) < 3: 
        true_coords = true_coords.unsqueeze(0)
    noised_coords = true_coords.clone()
    coords_scn = rearrange(true_coords, 'b (l c) d -> b l c d', c=14)

    ###### SETP 1: internals #########
    if NOISE_INTERNALS:
        # create noised and masked noised coords        
        noised_coords, cloud_mask = noise_internals(seq, angles = angles, 
                                                    coords = coords_scn.squeeze(),  
                                                    noise_scale = NOISE_INTERNALS, 
                                                    verbose = False)
        masked_noised = noised_coords[naive_cloud_mask]
        noised_coords = rearrange(noised_coords, 'l c d -> () (l c) d')

    ###### SETP 2: build from backbone #########
    if SIDECHAIN_RECONSTRUCT: 
        bb, mask = atom_selector(int_seq.unsqueeze(0), noised_coords, option="backbone", discard_absent=False)
        scaffolds = build_scaffolds_from_scn_angles(seq, angles=None, device="cpu")
        noised_coords[~mask] = 0.
        noised_coords = rearrange(noised_coords, '() (l c) d -> l c d', c=14)
        noised_coords, _ = sidechain_fold(wrapper = noised_coords.cpu(), **scaffolds, c_beta = False)
        noised_coords = rearrange(noised_coords, 'l c d -> () (l c) d').to(true_coords.device)


    return noised_coords, cloud_mask_flat



if __name__ == "__main__":
    import joblib
    # imports of data (from mp_nerf.utils.get_prot)
    prots = joblib.load("../../../segnn-pytorch/examples/custom_tests/infrastructure/prot_list_100_between_150_600.joblib")

    # set params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unpack and test
    seq, int_seq, true_coords, angles, padding_seq, mask, pid = prots[-1]

    true_coords = true_coords.unsqueeze(0)

    # check noised internals
    coords_scn = rearrange(true_coords, 'b (l c) d -> b l c d', c=14)
    cloud, cloud_mask = noise_internals(seq, angles=angles, coords=coords_scn[0], noise_scale=1.)
    print("cloud.shape", cloud.shape)

    # check integral
    integral, mask = combine_noise(true_coords, seq=seq, int_seq = None, angles=None,
                                   NOISE_INTERNALS=1e-2, SIDECHAIN_RECONSTRUCT=True)
    print("integral.shape", integral.shape)

    integral, mask = combine_noise(true_coords, seq=None, int_seq = int_seq, angles=None,
                                   NOISE_INTERNALS=1e-2, SIDECHAIN_RECONSTRUCT=True)
    print("integral.shape2", integral.shape)



