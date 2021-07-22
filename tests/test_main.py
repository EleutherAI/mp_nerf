import numpy as np
import torch

from mp_nerf import *
from mp_nerf.utils import *
from mp_nerf.kb_proteins import *
from mp_nerf.proteins import *

def test_nerf_and_dihedral():
    # create points
    a = torch.tensor([1,2,3]).float()
    b = torch.tensor([1,4,5]).float()
    c = torch.tensor([1,4,7]).float()
    d = torch.tensor([1,8,8]).float()
    # calculate internal references
    v1 = (b-a).numpy()
    v2 = (c-b).numpy()
    v3 = (d-c).numpy()
    # get angles
    theta = np.arccos( np.dot(v2, v3) / \
                      (np.linalg.norm(v2) * np.linalg.norm(v3) )) 

    normal_p  = np.cross(v1, v2) 
    normal_p_ = np.cross(v2, v3)
    chi = np.arccos( np.dot(normal_p, normal_p_) / \
                    (np.linalg.norm(normal_p) * np.linalg.norm(normal_p_) ))
    # get length:
    l = torch.tensor(np.linalg.norm(v3))
    theta = torch.tensor(theta)
    chi = torch.tensor(chi)
    # reconstruct
    # doesnt work because the scn angle was not measured correctly
    # so the method corrects that incorrection
    assert (mp_nerf_torch(a, b, c, l, theta, chi - np.pi) - torch.tensor([1,0,6])).sum().abs() < 0.1
    assert get_dihedral(a, b, c, d).item() == chi


def test_modify_angles_mask_with_torsions():
    # create inputs
    seq = "AGHHKLHRTVNMSTIL"
    angles_mask = torch.randn(2, 16, 14)
    torsions = torch.ones(16, 4)
    # ensure shape
    assert modify_angles_mask_with_torsions(seq, angles_mask, torsions).shape == angles_mask.shape, \
           "Shapes don't match"