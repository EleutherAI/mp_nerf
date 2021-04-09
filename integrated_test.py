##########################
# Clone repos with utils #
##########################

# !git clone https://github.com/hypnopump/geometric-vector-perceptron

import os
import sys
import time
import timeit
import logging
sys.path.append("../../geometric-vector-perceptron/examples")

# science
import numpy as np 
import torch
import sidechainnet
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB
VOCAB = VOCAB()

# process
import joblib

# custom
from massive_pnerf import *
import data_handler as geom_handler
import data_utils as geom_utils 

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
    dataloaders_ = sidechainnet.load(casp_version=7, with_pytorch="dataloaders")
    logger.info("Data has been loaded"+"\n"+sep)

    lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    stored  = [ geom_utils.get_prot(dataloader_=dataloaders_,
                                  vocab_=VOCAB, 
                                  min_len=desired_len+10,
                                  max_len=desired_len+50 + int(desired_len>500)*(desired_len-500), 
                                  verbose=1) for desired_len in lengths ]

    logger.info("Assessing lengths of: "+str([len(x[0]) for x in stored])+"\n")

    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for device in ["cpu", cuda_device]:

        logger.info("Preparing speed tests: for device "+repr(device)+"\n")

        for i,desired_len in enumerate(lengths):

            seq, true_coords, angles, padding_seq, mask, pid = stored[i]
            scaffolds = geom_handler.build_scaffolds_from_scn_angles(seq[:-padding_seq or None],
                                                                     angles[:-padding_seq or None].to(device))
            arguments = {"seq": seq[:-padding_seq or None], "device": device}
            arguments.update(scaffolds)

            logger.info("Assessing the speed of folding algorithm at length "+str(len(seq)-padding_seq)+"\n")

            logger.info( str( timeit.timeit('proto_fold(seq=seq[:-padding_seq or None], **scaffolds, device=device)',
            	                             globals=globals(), number=1000) )+" for 1000 calls" )

            logger.info("Saving the related information at {0}{1}_info.joblib\n".format(
                        BASE_FOLDER, desired_len))
            joblib.dump({"seq": seq, 
                         "true_coords": true_coords,
                         "angles": angles,
                         "padding_seq": padding_seq,
                         "mask": mask,
                         "pid": pid}, BASE_FOLDER+str(desired_len)+"_info.joblib")
            logger.info(sep)

    logger.info("Execution has finished\n")


























