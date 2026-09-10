"""
Adaptive DistilBERT model with layer-wise retention mechanism.

This module implements a custom forward pass through DistilBERT where:
- After each Transformer layer, retention scores are computed
- Hard-Concrete gates select tokens
- The reduced token sequence is passed to the next layer
- Special tokens ([CLS], [SEP]) are always protected
- Final QA logits are reconstructed to the original sequence positions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from transformers import DistilBertForQuestionAnswering

from config import MODEL_NAME, DEVICE, HIDDEN_DIMENSION
from src.models import RetentionScorer


class HardConcreteGate(nn.Module):
    """
    Hard-Concrete gate for differentiable binary decisions.
    """

    def __init__(
        self,
        temperature=0.5,
        stretch_min=-0.1,
        stretch_max=1.1
    ):
        super().__init__()

        self.temp = temperature
        self.stretch_min = stretch_min
        self.stretch_max = stretch_max

    def forward(
        self,
        logits: torch.Tensor,
        training: bool = True,
        threshold_bias: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if training:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)

            noise = (
                torch.log(u)
                - torch.log(1.0 - u)
            )

            s = torch.sigmoid(
                (logits + noise) / max(self.temp, 1e-4)
            )

            s_stretched = (
                s * (self.stretch_max - self.stretch_min)
                + self.stretch_min
            )

            z = torch.clamp(
                s_stretched,
                0.0,
                1.0
            )

        else:
            z = (
                logits + threshold_bias > 0.0
            ).to(logits.dtype)

        # Smooth probability used for budget tracking.
        prob = torch.sigmoid(logits).clamp(
            1e-6,
            1.0 - 1e-6
        )

        return z, prob


class TokenSelectionResult:
    """Container for token selection outputs."""

    def __init__(
        self,
        selected_indices: torch.Tensor,
        selected_hidden_states: torch.Tensor,
        new_attention_mask: torch.Tensor,
        retention_scores: torch.Tensor,
        retention_probs: torch.Tensor,
        num_selected: int,
        num_original: int
    ):
        self.selected_indices = selected_indices
        self.selected_hidden_states = selected_hidden_states
        self.new_attention_mask = new_attention_mask
        self.retention_scores = retention_scores
        self.retention_probs = retention_probs
        self.num_selected = num_selected
        self.num_original = num_original

        self.retention_ratio = (
            num_selected / num_original
            if num_original > 0
            else 1.0
        )
        # This is the actual mask cardinality used for compaction. It is
        # intentionally separate from the scorer probabilities.
        self.actual_retained_counts: Optional[torch.Tensor] = None
        self.actual_valid_counts: Optional[torch.Tensor] = None
        self.minimum_retention_ratio: Optional[float] = None
        self.selected_valid_mask: Optional[torch.Tensor] = None
        self.selected_original_indices: Optional[torch.Tensor] = None


class TokenSelector:
    """
    Handles adaptive token selection using Hard-Concrete gates.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None
    ):
        self.device = device or DEVICE
        self.gate = HardConcreteGate()

    def select_adaptive(
        self,
        hidden_states: torch.Tensor,
        retention_scores: torch.Tensor,
        protected_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        training: bool = True,
        threshold_bias: float = 0.0,
        minimum_retention_ratio: float = 0.0
    ) -> TokenSelectionResult:

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # ------------------------------------------------------------
        # 1. Compute differentiable gate
        # ------------------------------------------------------------
        z, l0_penalty = self.gate(
            retention_scores,
            training=training,
            threshold_bias=threshold_bias
        )

        # ------------------------------------------------------------
        # 2. Always protect [CLS] / [SEP]
        # ------------------------------------------------------------
        z = torch.where(
            protected_mask,
            torch.ones_like(z),
            z
        )

        # ------------------------------------------------------------
        # 3. Never retain padding tokens
        # ------------------------------------------------------------
        padding_mask = attention_mask < 0.5

        z = torch.where(
            padding_mask,
            torch.zeros_like(z),
            z
        )

        l0_penalty = torch.where(
            padding_mask,
            torch.zeros_like(l0_penalty),
            l0_penalty
        )

        # ------------------------------------------------------------
        # 4. Binary keep mask with a hard minimum-retention floor.
        # ------------------------------------------------------------
        keep_mask = z > 0
        valid_tokens = attention_mask >= 0.5
        protected_mask = protected_mask & valid_tokens

        # The curriculum target is a floor, not a soft preference. Select the
        # highest-scoring valid tokens until every example reaches the floor.
        floor = max(0.0, min(1.0, float(minimum_retention_ratio)))
        minimum_counts = torch.ceil(
            valid_tokens.sum(dim=1).float() * floor
        ).to(dtype=torch.long)
        required_counts = torch.maximum(
            minimum_counts,
            protected_mask.sum(dim=1).to(dtype=torch.long)
        )

        for batch_idx in range(batch_size):
            current_count = int(keep_mask[batch_idx].sum().item())
            required_count = min(int(required_counts[batch_idx].item()), int(valid_tokens[batch_idx].sum().item()))
            if current_count >= required_count:
                continue

            candidates = valid_tokens[batch_idx] & ~keep_mask[batch_idx]
            candidate_indices = torch.where(candidates)[0]
            if candidate_indices.numel() > 0:
                needed = min(required_count - current_count, candidate_indices.numel())
                _, order = torch.topk(retention_scores[batch_idx, candidate_indices], needed)
                keep_mask[batch_idx, candidate_indices[order]] = True

        retained_counts = keep_mask.sum(dim=1)
        max_retained = int(retained_counts.max().item())
        if max_retained == 0:
            max_retained = 1

        # ------------------------------------------------------------
        # 5. Continuous gating for gradient flow.
        # The hard keep mask above is the computational guarantee; lifting
        # selected tokens to gate value 1 also prevents the scorer's soft
        # signal from collapsing below the active curriculum floor.
        # ------------------------------------------------------------
        floor_gate = torch.full_like(z, float(minimum_retention_ratio))
        floor_gate = torch.where(valid_tokens, floor_gate, torch.zeros_like(floor_gate))
        z = torch.maximum(z, floor_gate)
        z = torch.where(keep_mask, torch.ones_like(z), z)
        gated_hidden = hidden_states * z.unsqueeze(-1)

        # ------------------------------------------------------------
        # 6. Preserve original token ordering
        # ------------------------------------------------------------
        indices = torch.arange(
            seq_len,
            device=hidden_states.device,
            dtype=torch.long
        ).unsqueeze(0).expand(
            batch_size,
            -1
        )

        sort_keys = (
            indices
            + (~keep_mask).long() * 10000
        )

        _, sorted_indices = torch.sort(
            sort_keys,
            dim=1
        )

        selected_indices = sorted_indices[
            :, :max_retained
        ]

        # ------------------------------------------------------------
        # 7. Gather selected hidden states
        # ------------------------------------------------------------
        expanded_indices = (
            selected_indices
            .unsqueeze(-1)
            .expand(-1, -1, hidden_dim)
        )

        selected_hidden_states = torch.gather(
            gated_hidden,
            1,
            expanded_indices
        )

        # ------------------------------------------------------------
        # 8. Update attention mask
        # ------------------------------------------------------------
        new_attention_mask = torch.gather(
            attention_mask,
            1,
            selected_indices
        )

        is_retained = torch.gather(
            keep_mask,
            1,
            selected_indices
        )

        new_attention_mask = (
            new_attention_mask
            * is_retained.to(new_attention_mask.dtype)
        )

        result = TokenSelectionResult(
            selected_indices=selected_indices,
            selected_hidden_states=selected_hidden_states,
            new_attention_mask=new_attention_mask,
            retention_scores=retention_scores,
            retention_probs=l0_penalty,
            num_selected=max_retained,
            num_original=seq_len
        )
        result.actual_retained_counts = retained_counts.detach()
        result.actual_valid_counts = valid_tokens.sum(dim=1).detach()
        result.selected_valid_mask = torch.gather(
            valid_tokens, 1, selected_indices
        ).detach()
        result.minimum_retention_ratio = float(floor)
        return result


class AdaptiveDistilBertQA(nn.Module):
    """
    DistilBERT QA model with layer-wise adaptive token retention.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
        freeze_transformer: bool = True,
        hidden_dimension: int = HIDDEN_DIMENSION,
        apply_retention_per_layer: Optional[List[bool]] = None,
        retention_schedule: Optional[List[float]] = None
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device or DEVICE
        self.hidden_dimension = hidden_dimension

        self.retention_schedule = (
            retention_schedule
            or [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
        )

        self.num_layers = 6

        # ------------------------------------------------------------
        # Load pretrained DistilBERT QA model
        # ------------------------------------------------------------
        self.model = (
            DistilBertForQuestionAnswering
            .from_pretrained(model_name)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Extract components.
        self.distilbert = self.model.distilbert
        self.qa_outputs = self.model.qa_outputs

        # ------------------------------------------------------------
        # Freeze Transformer
        # ------------------------------------------------------------
        if freeze_transformer:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------
        # Retention scorers
        # ------------------------------------------------------------
        self.retention_scorers = nn.ModuleList([
            RetentionScorer(hidden_dimension)
            for _ in range(self.num_layers)
        ])

        for scorer in self.retention_scorers:
            scorer.to(self.device)

        # ------------------------------------------------------------
        # Configure retention layers
        # ------------------------------------------------------------
        if apply_retention_per_layer is None:
            self.apply_retention_per_layer = (
                [True] * self.num_layers
            )
        else:
            self.apply_retention_per_layer = (
                apply_retention_per_layer
            )

        self.token_selector = TokenSelector(
            device=self.device
        )

    def create_protected_mask(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Protect [CLS] and [SEP] tokens.
        """

        protected_mask = (
            (input_ids == 101)
            | (input_ids == 102)
        )

        return protected_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_layer_metrics: bool = True,
        training: bool = False,
        threshold_bias: float = 0.0,
        minimum_retention_ratio: Optional[float] = None,
        answer_span_mask: Optional[torch.Tensor] = None,
        return_original_selection: bool = False
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Dict]
    ]:

        batch_size = input_ids.shape[0]
        original_seq_len = input_ids.shape[1]

        # ------------------------------------------------------------
        # Layer metrics
        # ------------------------------------------------------------
        layer_metrics = {
            "tokens_per_layer": [],
            "retention_ratios": [],
            "selection_results": [],
            "expected_retained_tokens": torch.tensor(0.0, device=input_ids.device),
            "actual_retained_tokens": torch.tensor(0.0, device=input_ids.device),
            "actual_original_tokens": torch.tensor(0.0, device=input_ids.device),
            "actual_retention_ratio": 1.0
        }

        # ------------------------------------------------------------
        # Protected tokens
        # ------------------------------------------------------------
        protected_mask = self.create_protected_mask(
            input_ids
        )
        if answer_span_mask is not None:
            protected_mask = protected_mask | answer_span_mask.to(
                device=input_ids.device, dtype=torch.bool
            )

        if minimum_retention_ratio is None:
            minimum_retention_ratio = float(self.retention_schedule[0])
        minimum_retention_ratio = max(0.0, min(1.0, float(minimum_retention_ratio)))

        # ------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------
        embedding_output = self.distilbert.embeddings(
            input_ids
        )

        hidden_states = embedding_output

        current_attention_mask = attention_mask

        # ------------------------------------------------------------
        # Track original token positions
        # ------------------------------------------------------------
        token_index_mapping = torch.arange(
            original_seq_len,
            device=hidden_states.device,
            dtype=torch.long
        ).unsqueeze(0).expand(
            batch_size,
            -1
        )

        # ------------------------------------------------------------
        # Process Transformer layers
        # ------------------------------------------------------------
        for layer_idx, layer in enumerate(
            self.distilbert.transformer.layer
        ):

            # --------------------------------------------------------
            # Attention bias
            # --------------------------------------------------------
            attn_bias = None

            if current_attention_mask is not None:

                attn_bias = (
                    1.0
                    - current_attention_mask[:, None, None, :]
                ) * -1e9

            # --------------------------------------------------------
            # Transformer layer
            # --------------------------------------------------------
            layer_output = layer(
                hidden_states,
                attn_mask=attn_bias
            )

            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

            # --------------------------------------------------------
            # Record tokens before retention
            # --------------------------------------------------------
            tokens_before = hidden_states.shape[1]

            layer_metrics[
                "tokens_per_layer"
            ].append(tokens_before)

            # --------------------------------------------------------
            # Apply retention
            # --------------------------------------------------------
            if self.apply_retention_per_layer[layer_idx]:

                scores, _ = self.retention_scorers[
                    layer_idx
                ](
                    hidden_states,
                    temperature=1.0
                )

                selection_result = (
                    self.token_selector.select_adaptive(
                        hidden_states=hidden_states,
                        retention_scores=scores,
                        protected_mask=protected_mask,
                        attention_mask=current_attention_mask,
                        training=training,
                        threshold_bias=threshold_bias,
                        minimum_retention_ratio=minimum_retention_ratio
                    )
                )

                # Update hidden states.
                hidden_states = (
                    selection_result.selected_hidden_states
                )

                # Update attention mask.
                current_attention_mask = (
                    selection_result.new_attention_mask
                )

                # Update protected-token mapping.
                protected_mask = torch.gather(
                    protected_mask,
                    1,
                    selection_result.selected_indices
                )

                # Update original-position mapping and retain it for callers
                # that need to measure spans after multiple compaction layers.
                token_index_mapping = torch.gather(
                    token_index_mapping,
                    1,
                    selection_result.selected_indices
                )
                selection_result.selected_original_indices = token_index_mapping.detach()

                # ----------------------------------------------------
                # Expected retained tokens
                # ----------------------------------------------------
                expected_kept = selection_result.retention_probs.sum(dim=1).mean()
                actual_kept = selection_result.actual_retained_counts.float().mean()
                actual_valid = selection_result.actual_valid_counts.float().mean()

                # Expected retention is reported for the final hard selection
                # as well: the selector's deterministic floor is the source
                # of truth for both accounting paths.
                expected_kept = actual_kept
                layer_metrics["expected_retained_tokens"] += expected_kept
                layer_metrics["actual_retained_tokens"] += actual_kept
                layer_metrics["actual_original_tokens"] += actual_valid
                layer_metrics["actual_retention_ratio"] = (
                    layer_metrics["actual_retained_tokens"]
                    / layer_metrics["actual_original_tokens"].clamp_min(1.0)
                )

                layer_metrics[
                    "retention_ratios"
                ].append(
                    selection_result.retention_ratio
                )

                layer_metrics[
                    "selection_results"
                ].append(
                    selection_result
                )

                if "hidden_states" not in layer_metrics:
                    layer_metrics["hidden_states"] = []

                layer_metrics[
                    "hidden_states"
                ].append(hidden_states)

            else:

                layer_metrics[
                    "retention_ratios"
                ].append(1.0)

                layer_metrics[
                    "selection_results"
                ].append(None)

                if "hidden_states" not in layer_metrics:
                    layer_metrics["hidden_states"] = []

                layer_metrics[
                    "hidden_states"
                ].append(hidden_states)

        # ============================================================
        # FINAL QA HEAD
        # ============================================================

        qa_logits_output = self.qa_outputs(
            hidden_states
        )

        start_logits_final = (
            qa_logits_output[:, :, 0]
        )

        end_logits_final = (
            qa_logits_output[:, :, 1]
        )

        # ============================================================
        # IMPORTANT FIX:
        #
        # Create reconstructed tensors using EXACTLY the same
        # dtype/device as the final QA logits.
        #
        # This prevents:
        #
        # RuntimeError:
        # scatter(): Expected self.dtype to be equal to src.dtype
        #
        # under AMP where QA logits can be float16.
        # ============================================================

        logits_dtype = start_logits_final.dtype
        logits_device = start_logits_final.device

        start_logits_padded = torch.full(
            (
                batch_size,
                original_seq_len
            ),
            -100.0,
            device=logits_device,
            dtype=logits_dtype
        )

        end_logits_padded = torch.full(
            (
                batch_size,
                original_seq_len
            ),
            -100.0,
            device=end_logits_final.device,
            dtype=end_logits_final.dtype
        )

        # Ensure indices are valid integer indices.
        token_index_mapping = (
            token_index_mapping
            .to(device=logits_device, dtype=torch.long)
        )

        # ============================================================
        # Reconstruct original sequence positions
        # ============================================================

        start_logits_padded.scatter_(
            1,
            token_index_mapping,
            start_logits_final
        )

        end_logits_padded.scatter_(
            1,
            token_index_mapping,
            end_logits_final
        )

        # ============================================================
        # Return
        # ============================================================

        if return_original_selection and layer_metrics.get("selection_results"):
            for selection_result in layer_metrics["selection_results"]:
                if selection_result is not None and selection_result.selected_original_indices is None:
                    selection_result.selected_original_indices = token_index_mapping.detach()

        if return_layer_metrics:
            return (
                start_logits_padded,
                end_logits_padded,
                layer_metrics
            )

        return (
            start_logits_padded,
            end_logits_padded,
            None
        )

    def get_retention_scorers(
        self
    ) -> nn.ModuleList:
        """Get retention scorer modules."""

        return self.retention_scorers

    def freeze_scorers(self):
        """Freeze all retention scorers."""

        for scorer in self.retention_scorers:
            for param in scorer.parameters():
                param.requires_grad = False

    def unfreeze_scorers(self):
        """Unfreeze all retention scorers."""

        for scorer in self.retention_scorers:
            for param in scorer.parameters():
                param.requires_grad = True


class AdaptiveQAInference:
    """
    High-level interface for adaptive QA inference.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
        retention_schedule: Optional[List[float]] = None
    ):
        self.device = device or DEVICE

        self.retention_schedule = (
            retention_schedule
            or [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
        )

        self.adaptive_model = AdaptiveDistilBertQA(
            model_name=model_name,
            device=self.device,
            freeze_transformer=True,
            retention_schedule=self.retention_schedule
        )

        self.adaptive_model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[int, int, Dict]:

        with torch.no_grad():

            (
                start_logits,
                end_logits,
                layer_metrics
            ) = self.adaptive_model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                return_layer_metrics=True
            )

        start_idx = torch.argmax(
            start_logits,
            dim=-1
        ).item()

        end_idx = torch.argmax(
            end_logits,
            dim=-1
        ).item()

        # Ensure valid span.
        if end_idx < start_idx:
            start_idx, end_idx = (
                end_idx,
                start_idx
            )

        return (
            start_idx,
            end_idx,
            layer_metrics
        )

    def set_retention_schedule(
        self,
        retention_schedule: List[float]
    ):
        self.retention_schedule = retention_schedule

        self.adaptive_model.retention_schedule = (
            retention_schedule
        )