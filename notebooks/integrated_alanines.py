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

            # get data
            keys = ["angles_np", "coords_np", "observed_sequence"]
            chain = pr.parsePDB(datafile, chain=chain, model=1)
            parsed = sidechainnet.utils.measure.get_seq_coords_and_angles(chain)
            data = {k:v for k,v in zip(keys, parsed)}
            # get scaffs
            scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(data["observed_sequence"], 
                                                                         torch.from_numpy(data["angles_np"]).to(device))

            logger.info("Assessing the speed of folding algorithm at file "+filenames[i]+"\n")

            logger.info( str( timeit.timeit('mp_nerf.proteins.protein_fold(**scaffolds, device=device, hybrid=hybrid)',
            	                             globals=globals(), number=1000) )+" for 1000 calls" )

            logger.info("Done")
            logger.info(sep)

    logger.info("Execution has finished\n")