##########################
# Clone repos with utils #
##########################

# !git clone https://github.com/hypnopump/geometric-vector-perceptron

import os
import sys
import time
import timeit
import logging

# science
import numpy as np 
import torch
import prody as pr
import sidechainnet
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB
VOCAB = VOCAB()

# process
import joblib

# custom
import mp_nerf

BASE_FOLDER = "experiments/"

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
                    # datefmt='%m-%d %H:%M',
                    filename=BASE_FOLDER+"logs_experiment.txt",
                    filemode="a")
logger = logging.getLogger()
sep = "\n\n=======\n\n"


def get_scn_format(datafile, pdb_code="auto", chain="A", num=1): 
    """ Gets a PDB ID in sidechainet format
        Inputs: 
        * datafile: str. url to RCSB PDB or relative route in the files system.
        * pdb_code: str. 4-letter PDB code. if auto, will take first 4 
                    letters of datafile file name. 
        * chain: str. the chain in the PDB structure to retrieve
        * num: int. the num in the PDB structure to retrieve. 
        Outputs: dict with
        * seq: 1-letter code string
        * coords (L, 14, 3)
        * mask (whether coords for residue are known or now)
    """
    data = {"seq": None, "coords": None, "mask": None}
    # get pdb code
    if pdb_code == "auto": 
        pdb_code = datafile.split("/")[-1][:4].upper()
    # download if URL
    download = False
    if datafile.startswith("http"):
        os.system("wget {0}".format(datafile))
        download = True
        datafile = datafile.split("/")[-1]
        
    # read data
    chain = pr.parsePDB(datafile, chain=chain, model=num)
    if download: 
        os.system("rm {0}".format(datafile))
    # download seq from PDB API
    # data["seq_pdb"] = sidechainnet.utils.download.get_seq_from_pdb(pdb_code, chain=1)
    # get coords
    keys = ["angles_np", "coords_np", "observed_sequence", "unmodified_sequence", "is_nonstd"]
    parsed = sidechainnet.utils.measure.get_seq_coords_and_angles(chain)
    return {k:v for k,v in zip(keys, parsed)}


# begin tests
if __name__ == "__main__":

    logger.info("Loading data"+"\n")
    
    dataloc = "experiments_manual/rclab_data/"
    filenames = [dataloc+x for x in os.listdir(dataloc) if x.endswith(".pdb")]

    run_opts = [(torch.device("cpu"), False)] # tuples of (device, hybrid)
    # add possibility for different configs
    if torch.cuda.is_available():
        run_opts.append( (torch.device("cuda"), True))
        run_opts.append( (torch.device("cuda"), False))


    for device,hybrid in run_opts:

        logger.info("Preparing speed tests: for device "+repr(device)+" and hybrid_opt: "+str(hybrid)+"\n")

        for i,filename in enumerate(filenames):

            data = get_scn_format(filename, chain="A", num=1)
            scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(data["observed_sequence"], 
                                                                         torch.from_numpy(data["angles_np"]).to(device))

            logger.info("Assessing the speed of folding algorithm at file "+filenames[i]+"\n")

            logger.info( str( timeit.timeit('mp_nerf.proteins.protein_fold(**scaffolds, device=device, hybrid=hybrid)',
            	                             globals=globals(), number=1000) )+" for 1000 calls" )

            logger.info("Done")
            logger.info(sep)

    logger.info("Execution has finished\n")