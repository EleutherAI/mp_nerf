<<<<<<< HEAD
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
    lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900]# [::-1]
    try: 
        "a"+9
        # skip
        dataloaders_ = sidechainnet.load(casp_version=7, with_pytorch="dataloaders", batch_size=2)
        logger.info("Data has been loaded"+"\n"+sep)
        stored  = [ mp_nerf.utils.get_prot(dataloader_=dataloaders_, 
                                           vocab_=VOCAB, 
                                           min_len=desired_len+5, 
                                           max_len=desired_len+60) for desired_len in lengths ]
        joblib.dump(stored, BASE_FOLDER[:-1]+"_manual/analyzed_prots.joblib")
    except: 
        stored = joblib.load(BASE_FOLDER[:-1]+"_manual/analyzed_prots.joblib")
        logger.info("Data has been loaded"+"\n"+sep)

    logger.info("Assessing lengths of: "+str([len(x[0]) for x in stored])+"\n")

    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for device in ["cpu", cuda_device]:

        logger.info("Preparing speed tests: for device "+repr(device)+"\n")

        for i,desired_len in enumerate(lengths):

            seq, int_seq, true_coords, angles, padding_seq, mask, pid = stored[i]
            scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq, angles.to(device))

            logger.info("Assessing the speed of folding algorithm at length "+str(len(seq))+"\n")

            logger.info( str( timeit.timeit('mp_nerf.proteins.protein_fold(**scaffolds, device=device, hybrid=True)',
            	                             globals=globals(), number=1000) )+" for 1000 calls" )

            logger.info("Saving the related information at {0}{1}_info.joblib\n".format(
                        BASE_FOLDER, desired_len))
            joblib.dump({"seq": seq, 
                         "true_coords": true_coords,
                         "angles": angles,
                         "padding_seq": padding_seq,
                         "mask": mask,
                         "pid": pid, 
                         "padding_stripped": True}, BASE_FOLDER+str(desired_len)+"_info.joblib")
            logger.info(sep)

    logger.info("Execution has finished\n")






=======
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
    "a"+9
    logger.info("Loading data"+"\n")
    # adapt desired lengths to previous experiments results
    # lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900]# [::-1]
    lengths = [[131, 150],
               [200, 250],
               [331, 351],
               [400, 450],
               [500, 550],
               [600, 650],
               [700, 780],
               [800, 900],
               [905, 1070]]# [::-1]
    try: 
        # skip
        logger.info("Data has been loaded"+"\n"+sep)
        stored  = [ ]
        for i,(desired_len, upper_len) in enumerate(lengths[::-1]): 
            dataloaders_ = sidechainnet.load(casp_version=7, with_pytorch="dataloaders", batch_size=2)
            stored.append(mp_nerf.utils.get_prot(dataloader_=dataloaders_, 
                                                 vocab_=VOCAB, 
                                                 min_len=desired_len, 
                                                 max_len=upper_len))
        stored = stored[::-1]

        joblib.dump(stored, BASE_FOLDER[:-1]+"_manual/analyzed_prots.joblib")
    except: 
        stored = joblib.load(BASE_FOLDER[:-1]+"_manual/analyzed_prots.joblib")
        logger.info("Data has been loaded"+"\n"+sep)

    logger.info("Assessing lengths of: "+str([len(x[0]) for x in stored])+"\n")

    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for device in ["cpu", cuda_device]:

        logger.info("Preparing speed tests: for device "+repr(device)+"\n")

        for i,desired_len in enumerate(lengths):

            seq, int_seq, true_coords, angles, padding_seq, mask, pid = stored[i]
            scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq, angles.to(device))

            logger.info("Assessing the speed of folding algorithm at length "+str(len(seq))+"\n")

            logger.info( str( timeit.timeit('mp_nerf.proteins.protein_fold(**scaffolds, device=device, hybrid=True)',
            	                             globals=globals(), number=1000) )+" for 1000 calls" )

            logger.info("Saving the related information at {0}{1}_info.joblib\n".format(
                        BASE_FOLDER, desired_len))
            joblib.dump({"seq": seq, 
                         "true_coords": true_coords,
                         "angles": angles,
                         "padding_seq": padding_seq,
                         "mask": mask,
                         "pid": pid, 
                         "padding_stripped": True}, BASE_FOLDER+str(desired_len)+"_info.joblib")
            logger.info(sep)

    logger.info("Execution has finished\n")






>>>>>>> 4cabbe55371d6a9a7edeab1db719fa0cf8312eae
