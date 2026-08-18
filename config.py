

import torch


MODEL_NAME = "distilbert-base-uncased-distilled-squad"


HIDDEN_DIMENSION = 768


NUM_TRANSFORMER_LAYERS = 6




RETENTION_RATIO = 0.50


TEMPERATURE = 1.0



SEED = 42


LEARNING_RATE = 1e-3


TRAINING_STEPS = 500


GRADIENT_CLIP = 1.0


OPTIMIZER_WEIGHT_DECAY = 1e-4



BUDGET_LAMBDA = 0.10


ENTROPY_LAMBDA = 0.001



RETENTION_SCORER_DROPOUT = 0.10


RETENTION_SCORER_INTERMEDIATE_DIM_RATIO = 4



def get_device():
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()




CHECKPOINT_PATH = "checkpoints/retention_scorer.pt"


LOG_INTERVAL = 25



QA_DATASET = "squad"


EVAL_BATCH_SIZE = 1



MAX_SEQUENCE_LENGTH = 512


TRUNCATION_ENABLED = True


ENABLE_DETAILED_LOGGING = True


HEADER_WIDTH = 80

