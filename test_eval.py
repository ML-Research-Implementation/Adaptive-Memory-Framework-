import traceback
from evaluate_squad import *
from src.models_adaptive import AdaptiveDistilBertQA
from transformers import AutoTokenizer

ammr = AdaptiveDistilBertQA()
ammr.retention_schedule = [0.9] * 6
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(batch_size=8, max_train_samples=10, max_val_samples=128)

try:
    evaluate_model(ammr, val_dl, val_features, val_data, tokenizer, is_baseline=False)
except Exception as e:
    traceback.print_exc()
