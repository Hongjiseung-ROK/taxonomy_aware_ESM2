import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig
from typing import Optional
from peft import get_peft_model, LoraConfig, TaskType

class TaxonomyAwareESM(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="facebook/esm2_t33_650M_UR50D", use_lora=True, lora_rank=8, vocab_sizes=None):
        super().__init__()
        
        # 1. ESM Backbone
        print(f"Loading ESM model: {pretrained_model_name}")
        self.esm = EsmModel.from_pretrained(pretrained_model_name)
        
        if use_lora:
            print(f"Injecting LoRA adapters (Rank={lora_rank})...")
            # Define configuration
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=lora_rank,            # Intrinsic Rank
                lora_alpha=32,          # Scaling Factor
                lora_dropout=0.1,
                target_modules=["query", "value"] 
            )
            # Wrap the model
            self.esm = get_peft_model(self.esm, peft_config)
            self.esm.print_trainable_parameters()
        else:
            # Traditional Freezing
            for param in self.esm.parameters():
                param.requires_grad = False
        
        self.hidden_dim = self.esm.config.hidden_size # e.g. 1280 for 650M
        
        # 2. Taxonomy Encoder
        # Ranks: Phylum, Class, Order, Family, Genus, Species, Subspecies (7 levels)
        if vocab_sizes is None:
            # Fallback to defaults if not provided (though train.py should pass them)
            # Using sufficiently large defaults relative to known biology data
            print("Warning: vocab_sizes not provided, using hardcoded defaults.")
            self.vocab_sizes = [1000, 5000, 10000, 20000, 50000, 100000, 50000]
        else:
            self.vocab_sizes = vocab_sizes
            print(f"Model initialized with vocab sizes: {self.vocab_sizes}")
        
        self.tax_embedding_dim = 128 # Dimension for tax embeddings
        self.tax_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_sizes[i], self.tax_embedding_dim, padding_idx=0) 
            for i in range(len(self.vocab_sizes))
        ])
        
        # Project combined tax embeddings (7 * 128) to match ESM dim?
        # Or project ESM to match Tax?
        # Goal: Cross Attention using Sequence as Query, Tax as Key/Value.
        # Sequence (B, L, D_esm)
        # Tax (B, 7, D_tax)
        
        self.tax_projector = nn.Linear(self.tax_embedding_dim, self.hidden_dim)
        
        # 3. Cross Attention Fusion
        # Query: Protein Sequence (B, L, H)
        # Key/Value: Taxonomy Info (B, 7, H)
        # Output: Enhanced Sequence (B, L, H)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 4. Classifier Head
        # Pooling? "Mean Pooling" or "CLS". ESM has CLS.
        # Let's use Mean Pooling of the Fused Sequence representation.
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, tax_vector):
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        tax_vector: (B, 7)
        """
        
        # 1. Forward ESM
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # (B, L, H)
        
        # 2. Forward Taxonomy
        # tax_vector is (B, 7) integers.
        # We need to embed each column with its specific embedding layer.
        tax_embeds = []
        for i in range(7):
            idx = tax_vector[:, i] # (B,)
            # Clamp to vocab size to avoid index out of bounds if new species appear?
            # Or assume 0 (UNK) handling upstream.
            # Ideally safety check:
            idx = idx.clamp(0, self.vocab_sizes[i] - 1)
            emb = self.tax_embeddings[i](idx) # (B, D_tax)
            tax_embeds.append(emb)
            
        # Stack to get (B, 7, D_tax)
        tax_sequence = torch.stack(tax_embeds, dim=1) # (B, 7, D_tax)
        
        # Project to match ESM dim
        tax_sequence = self.tax_projector(tax_sequence) # (B, 7, H)
        
        # 3. Cross Attention
        # Query: Sequence (B, L, H)
        # Key: Tax (B, 7, H)
        # Value: Tax (B, 7, H)
        # We want to inject Tax info into Sequence.
        
        attn_output, _ = self.cross_attention(
            query=sequence_output,
            key=tax_sequence,
            value=tax_sequence
        )
        
        # Residual Connection + Norm
        # fused_sequence = LayerNorm(Sequence + Dropout(Attn))
        fused_sequence = self.layer_norm(sequence_output + self.dropout(attn_output))
        
        # 4. Pooling
        # Masked Mean Pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
        sum_embeddings = torch.sum(fused_sequence * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask # (B, H)
        
        # 5. Classifier
        logits = self.classifier(pooled_output) # (B, NumClasses)
        
        return logits


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, reduction='mean', ic_weights: Optional[torch.Tensor] = None, normalize_ic: bool = True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction
        self.normalize_ic = normalize_ic

        # IC 가중치 처리
        self.use_ic_weights = ic_weights is not None
        if self.use_ic_weights:
            if normalize_ic:
                # 0이 아닌 가중치만 사용하여 정규화 (mean=1)
                nonzero_mask = ic_weights > 0
                if nonzero_mask.sum() > 0:
                    nonzero_weights = ic_weights[nonzero_mask]
                    ic_weights = ic_weights / (nonzero_weights.mean() + 1e-8)
            
            # 최소 가중치 설정 및 버퍼 등록
            ic_weights = torch.clamp(ic_weights, min=0.1)
            self.register_buffer('ic_weights', ic_weights)
            print("✅ AsymmetricLoss with IC weights initialized.")
            print(f"   IC weights range: [{self.ic_weights.min():.4f}, {self.ic_weights.max():.4f}]")

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos = one_sided_w * los_pos
            los_neg = one_sided_w * los_neg
        
        # IC 가중치 적용 (Positive와 Negative Loss 모두에)
        # 평가 시 FP에도 IC 가중치가 적용되므로, 학습 시에도 일관성을 위해 negative loss에도 적용
        if self.use_ic_weights:
            los_pos = los_pos * self.ic_weights.unsqueeze(0)
            los_neg = los_neg * self.ic_weights.unsqueeze(0)

        loss = - (los_pos + los_neg)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss

