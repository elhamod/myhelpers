#based on https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
# https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/
# https://www.mldawn.com/reproducibility-in-pytorch/ 

import os
import random
import numpy as np
import torch
import warnings

def seed(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    torch.cuda.manual_seed_all(seed_value)
    torch.cuda.manual_seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # g = torch.Generator()
    # g.manual_seed(seed_value)
    
    # torch.use_deterministic_algorithms(True) # raised an error because some things cant be deterministic

def get_seed_from_trialNumber(trialNumber):
    return trialNumber if trialNumber is not None else -1

# Deprecated.
def get_seed_from_Hex(trialHash):
    if trialHash is None:
        SEED_INT = -1
        warnings.warn("No random seed was passed!")
    else:
        SEED_INT = (int(trialHash, 16))%(2**31)

    return SEED_INT



# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, skip_top=50, seed=seed_value)
# model.add(Dropout(0.25, seed=seed_value))

# def _init_fn(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
# test_generator = torch.Generator()
# test_generator.manual_seed(SEED_INT)
# self.test_loader = torch.utils.data.DataLoader(self.dataset_test, pin_memory=True, generator=test_generator, shuffle=SHUFFLE, batch_size=batchSize, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn)