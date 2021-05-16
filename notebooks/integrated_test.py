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


def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, verbose=True):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training. 
        Inputs: 
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
        Outputs: (cleaned, without padding)
        (seq_str, int_seq, coords, angles, padding_seq, mask, pid)
    """
    for b,batch in enumerate(dataloader_['train']):
        # try for breaking from 2 loops at once
        try:
            for i in range(batch.int_seqs.shape[0]):
                # strip padding padding
                padding_seq = (batch.int_seqs[i] == 20).sum().item()
                padding_angles = (torch.abs(batch.angs[i]).sum(dim=-1) == 0).long().sum().item()

                if padding_seq == padding_angles:
                    # check for appropiate length
                    real_len = batch.int_seqs[i].shape[0] - padding_seq
                    if max_len >= real_len >= min_len:
                        # strip padding tokens
                        seq = ''.join([vocab_.int2char(aa) for aa in batch.int_seqs[i].numpy()])
                        seq = seq[:-padding_seq or None]
                        int_seq = batch.int_seqs[i][:-padding_seq or None]
                        angles  = batch.angs[i][:-padding_seq or None]
                        mask    = batch.msks[i][:-padding_seq or None]
                        coords  = batch.crds[i][:-padding_seq*14 or None]

                        print("stopping at sequence of length", real_len)
                        raise StopIteration
                else:
                    # print("found a seq of length:", len(seq),
                    #        "but oustide the threshold:", min_len, max_len)
                    pass

        except StopIteration:
            break
            
    return seq, int_seq, coords, angles, padding_seq, mask, batch.pids[i]

# begin tests
if __name__ == "__main__":

    logger.info("Loading data"+"\n")
    lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900]# [::-1]
    try: 
        "a"+9
        # skip
        dataloaders_ = sidechainnet.load(casp_version=7, with_pytorch="dataloaders", batch_size=2)
        logger.info("Data has been loaded"+"\n"+sep)
        stored  = [ get_prot(dataloader_=dataloaders_, 
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

            logger.info( str( timeit.timeit('mp_nerf.proteins.protein_fold(**scaffolds, device=device)',
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






