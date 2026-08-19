"""
Initialize the src package.
Exports commonly used modules and classes.
"""

from config import *
from src.utils import (
    set_seed,
    initialize_reproducibility,
    print_header,
    count_parameters,
    count_trainable_parameters,
    freeze_model,
    unfreeze_model,
    get_device,
)
from src.data import (
    QADataLoader,
    find_answer_span,
    get_token_type,
    create_token_masks,
)
from src.models import (
    RetentionScorer,
    SoftRetentionGate,
    AdaptiveMemoryRetention,
)
from src.models_adaptive import (
    AdaptiveDistilBertQA,
    AdaptiveQAInference,
    TokenSelector,
    TokenSelectionResult,
)
from src.metrics import (
    LayerWiseMetrics,
    LayerMetrics,
    SequenceMetrics,
    InferenceTimer,
)
from src.losses import (
    calculate_qa_loss,
    calculate_budget_loss,
    calculate_entropy_loss,
    calculate_combined_loss,
)
from src.baseline import (
    BaselineQAModel,
    compute_baseline_metrics,
)
from src.training import (
    RetentionScorerTrainer,
    train_retention_scorer,
)
from src.evaluation import (
    RetentionAnalyzer,
    get_top_k_tokens,
    compare_predictions,
    print_evaluation_report,
)

__all__ = [
    # Config
    'MODEL_NAME',
    'HIDDEN_DIMENSION',
    'RETENTION_RATIO',
    'TEMPERATURE',
    'LEARNING_RATE',
    'TRAINING_STEPS',
    'BUDGET_LAMBDA',
    'ENTROPY_LAMBDA',
    'DEVICE',
    # Utils
    'set_seed',
    'initialize_reproducibility',
    'print_header',
    'count_parameters',
    'count_trainable_parameters',
    'freeze_model',
    'unfreeze_model',
    'get_device',
    # Data
    'QADataLoader',
    'find_answer_span',
    'get_token_type',
    'create_token_masks',
    # Models
    'RetentionScorer',
    'SoftRetentionGate',
    'AdaptiveMemoryRetention',
    # Adaptive Models (Layer-wise Retention)
    'AdaptiveDistilBertQA',
    'AdaptiveQAInference',
    'TokenSelector',
    'TokenSelectionResult',
    # Metrics
    'LayerWiseMetrics',
    'LayerMetrics',
    'SequenceMetrics',
    'InferenceTimer',
    # Losses
    'calculate_qa_loss',
    'calculate_budget_loss',
    'calculate_entropy_loss',
    'calculate_combined_loss',
    # Baseline
    'BaselineQAModel',
    'compute_baseline_metrics',
    # Training
    'RetentionScorerTrainer',
    'train_retention_scorer',
    # Evaluation
    'RetentionAnalyzer',
    'get_top_k_tokens',
    'compare_predictions',
    'print_evaluation_report',
]
