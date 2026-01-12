"""
1D CNN 모델 아키텍처 (with Residual Connection)
================================================
단백질 기능 예측을 위한 1D CNN 기반 분류 모델

입력: (batch, seq_len, embedding_dim) -> 내부에서 (batch, embedding_dim, seq_len)으로 변환
Conv1d는 (batch, channels, length) 형태를 기대함
Skip Connection: 3개 이상의 레이어에서 입력을 출력에 더함

사용법:
    from model import Hierarchical1DCNNClassifier, ModelConfig
    
    # 방법 1: ModelConfig 사용
    config = ModelConfig(
        embedding_dim=1280,
        num_classes=40122,
        conv_channels=[512, 1024, 2048],
        kernel_sizes=[7, 5, 3],
        fc_dims=[2048],
        dropout=0.3,
        use_residual=True
    )
    model = Hierarchical1DCNNClassifier.from_config(config)
    
    # 방법 2: 직접 파라미터 전달
    model = Hierarchical1DCNNClassifier(
        embedding_dim=1280,
        num_classes=40122,
        conv_channels=[512, 1024, 2048],
        kernel_sizes=[7, 5, 3],
        fc_dims=[2048],
        dropout=0.3,
        use_residual=True
    )
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """
    1D CNN 모델 설정 클래스
    
    모든 하이퍼파라미터를 한 곳에서 관리하고, 실험 재현성을 보장합니다.
    
    Attributes:
        embedding_dim: 입력 임베딩 차원 (ESM2: 1280)
        num_classes: 출력 클래스 수 (GO Term 수)
        conv_channels: 각 Conv 레이어의 출력 채널 수 리스트
        kernel_sizes: 각 Conv 레이어의 커널 크기 리스트 (conv_channels와 동일한 길이)
        fc_dims: Fully Connected 레이어 차원 리스트
        dropout: Dropout 비율 (0.0 ~ 1.0)
        conv_dropout_ratio: Conv 레이어 Dropout 비율 (dropout의 배수, 기본 0.5)
        use_residual: Residual Connection 사용 여부
        min_layers_for_residual: Residual Connection을 활성화할 최소 레이어 수
        pooling_mode: Global Pooling 방식 ('max', 'avg', 'concat')
        use_batch_norm: BatchNorm 사용 여부
        activation: 활성화 함수 ('relu', 'gelu', 'leaky_relu')
    """
    # 필수 파라미터
    embedding_dim: int = 1280
    num_classes: int = 40122
    
    # Convolutional Layer 설정
    conv_channels: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    
    # Fully Connected Layer 설정
    fc_dims: List[int] = field(default_factory=lambda: [2048])
    
    # Regularization 설정
    dropout: float = 0.3
    conv_dropout_ratio: float = 0.5  # conv layer dropout = dropout * conv_dropout_ratio
    
    # Residual Connection 설정
    use_residual: bool = True
    min_layers_for_residual: int = 3  # 이 수 이상의 레이어일 때만 residual 활성화
    
    # Advanced 설정
    pooling_mode: str = 'concat'  # 'max', 'avg', 'concat'
    use_batch_norm: bool = True
    activation: str = 'relu'  # 'relu', 'gelu', 'leaky_relu'
    
    def __post_init__(self):
        """설정 유효성 검사"""
        if len(self.conv_channels) != len(self.kernel_sizes):
            raise ValueError(
                f"conv_channels({len(self.conv_channels)})와 "
                f"kernel_sizes({len(self.kernel_sizes)})의 길이가 일치해야 합니다."
            )
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout은 0과 1 사이여야 합니다. 현재: {self.dropout}")
        
        if self.pooling_mode not in ['max', 'avg', 'concat']:
            raise ValueError(f"pooling_mode는 'max', 'avg', 'concat' 중 하나여야 합니다. 현재: {self.pooling_mode}")
        
        if self.activation not in ['relu', 'gelu', 'leaky_relu']:
            raise ValueError(f"activation은 'relu', 'gelu', 'leaky_relu' 중 하나여야 합니다. 현재: {self.activation}")
    
    @property
    def num_conv_layers(self) -> int:
        """Conv 레이어 수"""
        return len(self.conv_channels)
    
    @property
    def effective_use_residual(self) -> bool:
        """실제 Residual Connection 사용 여부 (레이어 수 조건 반영)"""
        return self.use_residual and (self.num_conv_layers >= self.min_layers_for_residual)
    
    @property
    def conv_dropout(self) -> float:
        """Conv 레이어 Dropout 비율"""
        return self.dropout * self.conv_dropout_ratio
    
    @property
    def pooled_dim(self) -> int:
        """Global Pooling 후 차원"""
        final_channels = self.conv_channels[-1]
        if self.pooling_mode == 'concat':
            return final_channels * 2
        return final_channels
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'fc_dims': self.fc_dims,
            'dropout': self.dropout,
            'conv_dropout_ratio': self.conv_dropout_ratio,
            'use_residual': self.use_residual,
            'min_layers_for_residual': self.min_layers_for_residual,
            'pooling_mode': self.pooling_mode,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)
    
    def summary(self) -> str:
        """설정 요약 출력"""
        lines = [
            "=" * 60,
            "📋 Model Configuration",
            "=" * 60,
            f"  입력 임베딩 차원: {self.embedding_dim}",
            f"  출력 클래스 수: {self.num_classes:,}",
            "",
            "  🔹 Convolutional Layers:",
            f"     레이어 수: {self.num_conv_layers}",
            f"     채널 수: {self.conv_channels}",
            f"     커널 크기: {self.kernel_sizes}",
            f"     Residual Connection: {'✅' if self.effective_use_residual else '❌'}",
            "",
            "  🔹 Fully Connected Layers:",
            f"     FC 차원: {self.fc_dims}",
            f"     Pooled 입력 차원: {self.pooled_dim}",
            "",
            "  🔹 Regularization:",
            f"     FC Dropout: {self.dropout}",
            f"     Conv Dropout: {self.conv_dropout:.3f}",
            "",
            "  🔹 Advanced:",
            f"     Pooling 모드: {self.pooling_mode}",
            f"     BatchNorm: {'✅' if self.use_batch_norm else '❌'}",
            f"     활성화 함수: {self.activation}",
            "=" * 60,
        ]
        return "\n".join(lines)


def get_activation(name: str) -> nn.Module:
    """활성화 함수 생성"""
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unknown activation: {name}")


class Conv1DBlock(nn.Module):
    """
    1D Convolution Block with BatchNorm, Activation, and optional Dropout
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        kernel_size: 커널 크기
        stride: 스트라이드 (기본: 1)
        padding: 패딩 ('same' 또는 정수, 기본: 'same')
        dropout: Dropout 비율 (기본: 0.1)
        use_batch_norm: BatchNorm 사용 여부 (기본: True)
        activation: 활성화 함수 이름 (기본: 'relu')
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: str = 'same', 
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(Conv1DBlock, self).__init__()
        
        # padding='same'을 수동 계산
        if padding == 'same':
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualConvBlock(nn.Module):
    """
    Residual Block: Conv1D + Skip Connection
    
    Skip Connection 로직:
    - 입력 채널 != 출력 채널: 1x1 Conv로 차원 맞춤
    - 입력 채널 == 출력 채널: Identity
    - output = Conv(input) + skip(input)
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        kernel_size: 커널 크기
        stride: 스트라이드 (기본: 1)
        dropout: Dropout 비율 (기본: 0.1)
        use_batch_norm: BatchNorm 사용 여부 (기본: True)
        activation: 활성화 함수 이름 (기본: 'relu')
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(ResidualConvBlock, self).__init__()
        
        padding = kernel_size // 2
        
        # Main path
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection (차원 맞춤)
        if in_channels != out_channels:
            skip_layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)]
            if use_batch_norm:
                skip_layers.append(nn.BatchNorm1d(out_channels))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        # Main path
        out = self.conv(x)
        out = self.bn(out)
        
        # Skip connection
        residual = self.skip(x)
        
        # Add & Activate
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class Hierarchical1DCNNClassifier(nn.Module):
    """
    1D CNN 기반 단백질 기능 예측 모델 (with Residual Connection)
    
    Architecture:
    1. Convolutional Feature Extraction
       - Multiple conv layers with increasing channels
       - Residual Connection (3개 이상 레이어 시 활성화)
       - BatchNorm + Activation + Dropout
    2. Global Pooling (Max + Avg concatenation)
    3. Fully Connected Layers for classification
    
    Usage:
        # 방법 1: 직접 파라미터 전달
        model = Hierarchical1DCNNClassifier(
            embedding_dim=1280,
            num_classes=40122,
            conv_channels=[512, 1024, 2048],
            kernel_sizes=[7, 5, 3],
            fc_dims=[2048],
            dropout=0.3,
            use_residual=True
        )
        
        # 방법 2: ModelConfig 사용
        config = ModelConfig(...)
        model = Hierarchical1DCNNClassifier.from_config(config)
    """
    
    def __init__(
        self, 
        embedding_dim: int = 1280, 
        num_classes: int = 40122, 
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        fc_dims: List[int] = None, 
        dropout: float = 0.3, 
        use_residual: bool = True,
        # Advanced parameters
        conv_dropout_ratio: float = 0.5,
        min_layers_for_residual: int = 3,
        pooling_mode: str = 'concat',
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Args:
            embedding_dim: ESM2 임베딩 차원 (1280)
            num_classes: GO Term 수
            conv_channels: 각 conv layer의 출력 채널 수 (기본: [512, 1024, 2048])
            kernel_sizes: 각 conv layer의 커널 크기 (기본: [7, 5, 3])
            fc_dims: FC layer 차원 (기본: [2048])
            dropout: Dropout 비율
            use_residual: Residual Connection 사용 여부
            conv_dropout_ratio: Conv 레이어 dropout = dropout * conv_dropout_ratio
            min_layers_for_residual: Residual을 활성화할 최소 레이어 수
            pooling_mode: 'max', 'avg', 'concat' 중 선택
            use_batch_norm: BatchNorm 사용 여부
            activation: 활성화 함수 ('relu', 'gelu', 'leaky_relu')
        """
        super(Hierarchical1DCNNClassifier, self).__init__()
        
        # 기본값 설정
        if conv_channels is None:
            conv_channels = [512, 1024, 2048]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if fc_dims is None:
            fc_dims = [2048]
        
        # 유효성 검사
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                f"conv_channels({len(conv_channels)})와 "
                f"kernel_sizes({len(kernel_sizes)})의 길이가 일치해야 합니다."
            )
        
        # 설정 저장
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_conv_layers = len(conv_channels)
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.fc_dims = fc_dims
        self.dropout = dropout
        self.pooling_mode = pooling_mode
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # 3개 이상의 레이어면 Residual Connection 사용
        self.use_residual = use_residual and (self.num_conv_layers >= min_layers_for_residual)
        
        # Conv 레이어 Dropout
        conv_dropout = dropout * conv_dropout_ratio
        
        # =====================
        # 1. Convolutional Layers
        # =====================
        self.conv_layers = nn.ModuleList()
        in_channels = embedding_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            if self.use_residual:
                # Residual Block 사용
                self.conv_layers.append(
                    ResidualConvBlock(
                        in_channels, out_channels, kernel_size, 
                        dropout=conv_dropout,
                        use_batch_norm=use_batch_norm,
                        activation=activation
                    )
                )
            else:
                # 일반 Conv Block 사용
                self.conv_layers.append(
                    Conv1DBlock(
                        in_channels, out_channels, kernel_size, 
                        dropout=conv_dropout,
                        use_batch_norm=use_batch_norm,
                        activation=activation
                    )
                )
            in_channels = out_channels
        
        # 마지막 conv layer의 출력 채널
        final_conv_channels = conv_channels[-1]
        
        # =====================
        # 2. Global Pooling
        # =====================
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Pooling 후 차원 계산
        if pooling_mode == 'concat':
            pooled_dim = final_conv_channels * 2  # max + avg 결합
        else:
            pooled_dim = final_conv_channels
        
        self.pooled_dim = pooled_dim
        
        # =====================
        # 3. Fully Connected Layers
        # =====================
        fc_layer_list = []
        prev_dim = pooled_dim
        
        for fc_dim in fc_dims:
            fc_layer_list.append(nn.Linear(prev_dim, fc_dim))
            if use_batch_norm:
                fc_layer_list.append(nn.BatchNorm1d(fc_dim))
            fc_layer_list.append(get_activation(activation))
            fc_layer_list.append(nn.Dropout(dropout))
            prev_dim = fc_dim
        
        # Output layer
        fc_layer_list.append(nn.Linear(prev_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layer_list)
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> 'Hierarchical1DCNNClassifier':
        """
        ModelConfig에서 모델 생성
        
        Args:
            config: ModelConfig 인스턴스
        
        Returns:
            Hierarchical1DCNNClassifier 인스턴스
        """
        return cls(
            embedding_dim=config.embedding_dim,
            num_classes=config.num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            use_residual=config.use_residual,
            conv_dropout_ratio=config.conv_dropout_ratio,
            min_layers_for_residual=config.min_layers_for_residual,
            pooling_mode=config.pooling_mode,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 입력 임베딩 (batch, seq_len, embedding_dim)
            attention_mask: 어텐션 마스크 (batch, seq_len) - 1=valid, 0=padding
            
        Returns:
            logits: (batch, num_classes)
        """
        # (batch, seq_len, embedding_dim) -> (batch, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Masking: 패딩 위치를 0으로 설정
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, seq_len)
            mask = attention_mask.unsqueeze(1)
            x = x * mask
        
        # Convolutional layers (ModuleList 순회)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)  # Residual or Plain Conv Block
        
        # Global pooling
        if self.pooling_mode == 'max':
            pooled = self.global_max_pool(x).squeeze(-1)
        elif self.pooling_mode == 'avg':
            pooled = self.global_avg_pool(x).squeeze(-1)
        else:  # concat
            max_pooled = self.global_max_pool(x).squeeze(-1)  # (batch, conv_channels[-1])
            avg_pooled = self.global_avg_pool(x).squeeze(-1)  # (batch, conv_channels[-1])
            pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # (batch, conv_channels[-1] * 2)
        
        # Fully connected layers
        logits = self.fc_layers(pooled)  # (batch, num_classes)
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """현재 모델 설정 반환"""
        return {
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'fc_dims': self.fc_dims,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'pooling_mode': self.pooling_mode,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def summary(self) -> str:
        """모델 요약 출력"""
        params = self.count_parameters()
        param_memory = params['total'] * 4 / (1024**2)  # float32 = 4 bytes
        
        lines = [
            "=" * 60,
            "📐 Model Summary: Hierarchical1DCNNClassifier",
            "=" * 60,
            "",
            "🔹 Architecture:",
            f"   Conv Layers: {self.num_conv_layers}",
            f"   Conv Channels: {self.conv_channels}",
            f"   Kernel Sizes: {self.kernel_sizes}",
            f"   FC Dims: {self.fc_dims}",
            f"   Residual Connection: {'✅' if self.use_residual else '❌'}",
            f"   Pooling Mode: {self.pooling_mode}",
            "",
            "🔹 Input/Output:",
            f"   Input: (batch, seq_len, {self.embedding_dim})",
            f"   Output: (batch, {self.num_classes})",
            "",
            "📊 Parameters:",
            f"   Total: {params['total']:,}",
            f"   Trainable: {params['trainable']:,}",
            f"   Memory (est.): {param_memory:.2f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)


# =========================================================================
# 편의 함수
# =========================================================================

def create_model(
    embedding_dim: int = 1280,
    num_classes: int = 40122,
    conv_channels: List[int] = None,
    kernel_sizes: List[int] = None,
    fc_dims: List[int] = None,
    dropout: float = 0.3,
    use_residual: bool = True,
    device: str = None,
    **kwargs
) -> Hierarchical1DCNNClassifier:
    """
    모델 생성 편의 함수
    
    Args:
        embedding_dim: 임베딩 차원
        num_classes: 클래스 수
        conv_channels: Conv 채널 리스트
        kernel_sizes: 커널 크기 리스트
        fc_dims: FC 차원 리스트
        dropout: Dropout 비율
        use_residual: Residual 사용 여부
        device: 디바이스 ('cuda', 'cpu', None=자동)
        **kwargs: 추가 파라미터
    
    Returns:
        모델 인스턴스 (지정된 디바이스로 이동됨)
    """
    model = Hierarchical1DCNNClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        fc_dims=fc_dims,
        dropout=dropout,
        use_residual=use_residual,
        **kwargs
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return model.to(device)


def test_model(model: Hierarchical1DCNNClassifier, seq_length: int = 1000, batch_size: int = 2):
    """
    모델 테스트 (더미 입력으로 forward pass 확인)
    
    Args:
        model: 테스트할 모델
        seq_length: 시퀀스 길이
        batch_size: 배치 크기
    """
    device = next(model.parameters()).device
    
    dummy_input = torch.randn(batch_size, seq_length, model.embedding_dim).to(device)
    dummy_mask = torch.ones(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        dummy_output = model(dummy_input, dummy_mask)
    
    print("🔍 모델 테스트:")
    print(f"   입력 shape: {dummy_input.shape}")
    print(f"   출력 shape: {dummy_output.shape}")
    print(f"   ✅ Forward pass 성공!")


# =========================================================================
# 학습 설정 클래스
# =========================================================================

@dataclass
class TrainingConfig:
    """
    학습 관련 모든 설정을 관리하는 클래스
    
    Attributes:
        # AMP 설정
        use_amp: Automatic Mixed Precision 사용 여부 (GPU 사용 시 권장)
        
        # Gradient Clipping
        grad_clip_norm: Gradient clipping max norm (학습 안정성)
        
        # 학습 파라미터
        num_epochs: 학습 에폭 수
        learning_rate: 학습률
        weight_decay: L2 정규화 가중치
        batch_size: 배치 크기
        
        # 스케줄러 설정
        scheduler_factor: LR 감소 비율
        scheduler_patience: 개선 없을 시 대기 에폭 수
        scheduler_min_lr: 최소 학습률
        
        # Cosine Annealing Warm Restarts 설정
        cosine_t_0: 첫 번째 restart까지 에폭 수
        cosine_t_mult: restart 주기 배수
        
        # Early Stopping
        early_stop_patience: Early stopping patience
        early_stop_min_delta: 최소 개선 기준
        
        # 기타
        use_pos_weight: 클래스 불균형 가중치 사용 여부
        non_blocking: Non-blocking data transfer 사용 여부
    """
    # AMP 설정
    use_amp: bool = None  # None이면 자동 감지 (GPU 있으면 True)
    
    # Gradient Clipping
    grad_clip_norm: float = 1.0
    
    # 학습 파라미터
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    
    # 스케줄러 설정
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-7
    
    # Cosine Annealing Warm Restarts 설정
    cosine_t_0: int = 10  # 첫 번째 restart까지 에폭 수
    cosine_t_mult: int = 2  # restart 주기 배수
    
    # Early Stopping
    early_stop_patience: int = 7
    early_stop_min_delta: float = 1e-4
    
    # 기타
    use_pos_weight: bool = False
    non_blocking: bool = True
    
    def __post_init__(self):
        """설정 초기화 후 처리"""
        # use_amp가 None이면 자동 감지
        if self.use_amp is None:
            self.use_amp = torch.cuda.is_available()
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'use_amp': self.use_amp,
            'grad_clip_norm': self.grad_clip_norm,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_min_lr': self.scheduler_min_lr,
            'cosine_t_0': self.cosine_t_0,
            'cosine_t_mult': self.cosine_t_mult,
            'early_stop_patience': self.early_stop_patience,
            'early_stop_min_delta': self.early_stop_min_delta,
            'use_pos_weight': self.use_pos_weight,
            'non_blocking': self.non_blocking,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)
    
    def summary(self) -> str:
        """설정 요약 출력"""
        lines = [
            "=" * 60,
            "⚡ Training Configuration",
            "=" * 60,
            "",
            "  🔹 AMP & Optimization:",
            f"     use_amp: {'✅' if self.use_amp else '❌'}",
            f"     grad_clip_norm: {self.grad_clip_norm}",
            f"     non_blocking: {'✅' if self.non_blocking else '❌'}",
            "",
            "  🔹 Training Parameters:",
            f"     num_epochs: {self.num_epochs}",
            f"     learning_rate: {self.learning_rate}",
            f"     weight_decay: {self.weight_decay}",
            f"     batch_size: {self.batch_size}",
            "",
            "  🔹 Scheduler:",
            f"     factor: {self.scheduler_factor}",
            f"     patience: {self.scheduler_patience}",
            f"     min_lr: {self.scheduler_min_lr}",
            "",
            "  🔹 Early Stopping:",
            f"     patience: {self.early_stop_patience}",
            f"     min_delta: {self.early_stop_min_delta}",
            "",
            "  🔹 Other:",
            f"     use_pos_weight: {'✅' if self.use_pos_weight else '❌'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =========================================================================
# 학습/검증/예측 함수 (고성능 최적화 버전)
# =========================================================================
# 최적화 기능:
# 1. AMP (Automatic Mixed Precision): 메모리 절약 + 속도 향상
# 2. Gradient Clipping: 학습 안정성 향상
# 3. Non-blocking Data Transfer: pin_memory와 함께 사용 시 속도 향상
# =========================================================================

# tqdm import (학습 진행률 표시용)
try:
    from tqdm import tqdm
except ImportError:
    # tqdm이 없으면 간단한 대체 함수 사용
    def tqdm(iterable, **kwargs):
        return iterable

# numpy import (predict 함수용)
try:
    import numpy as np
except ImportError:
    np = None


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
    config: TrainingConfig = None,
    use_amp: bool = None,
    grad_clip_norm: float = None,
    non_blocking: bool = None
) -> float:
    """
    한 에폭 학습 (AMP + Gradient Clipping + Non-blocking Transfer)
    
    Args:
        model: 학습할 모델
        dataloader: 학습 데이터로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 디바이스 (cuda/cpu)
        scaler: AMP GradScaler (None이면 자동 생성)
        config: TrainingConfig 인스턴스 (우선 적용)
        use_amp: AMP 사용 여부 (config가 없을 때 사용)
        grad_clip_norm: Gradient clipping norm (config가 없을 때 사용)
        non_blocking: Non-blocking transfer 사용 여부 (config가 없을 때 사용)
    
    Returns:
        avg_loss: 평균 손실
    """
    # 설정 결정 (config > 개별 파라미터 > 기본값)
    if config is not None:
        _use_amp = config.use_amp
        _grad_clip_norm = config.grad_clip_norm
        _non_blocking = config.non_blocking
    else:
        _use_amp = use_amp if use_amp is not None else torch.cuda.is_available()
        _grad_clip_norm = grad_clip_norm if grad_clip_norm is not None else 1.0
        _non_blocking = non_blocking if non_blocking is not None else True
    
    # GradScaler 생성 (필요시)
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=_use_amp)
    
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Non-blocking data transfer (pin_memory=True일 때 효과적)
        embeddings = batch['embedding'].to(device, non_blocking=_non_blocking)
        attention_mask = batch['attention_mask'].to(device, non_blocking=_non_blocking)
        labels = batch['label'].to(device, non_blocking=_non_blocking)
        
        optimizer.zero_grad()
        
        # AMP autocast (Mixed Precision)
        with torch.amp.autocast('cuda', enabled=_use_amp):
            logits = model(embeddings, attention_mask)
            loss = criterion(logits, labels)
        
        # AMP backward + Gradient Clipping
        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # Gradient Clipping (학습 안정성)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip_norm)
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig = None,
    use_amp: bool = None,
    non_blocking: bool = None
) -> float:
    """
    한 에폭 검증 (AMP + Non-blocking Transfer)
    
    Args:
        model: 검증할 모델
        dataloader: 검증 데이터로더
        criterion: 손실 함수
        device: 디바이스 (cuda/cpu)
        config: TrainingConfig 인스턴스 (우선 적용)
        use_amp: AMP 사용 여부 (config가 없을 때 사용)
        non_blocking: Non-blocking transfer 사용 여부 (config가 없을 때 사용)
    
    Returns:
        avg_loss: 평균 손실
    """
    # 설정 결정
    if config is not None:
        _use_amp = config.use_amp
        _non_blocking = config.non_blocking
    else:
        _use_amp = use_amp if use_amp is not None else torch.cuda.is_available()
        _non_blocking = non_blocking if non_blocking is not None else True
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            # Non-blocking data transfer
            embeddings = batch['embedding'].to(device, non_blocking=_non_blocking)
            attention_mask = batch['attention_mask'].to(device, non_blocking=_non_blocking)
            labels = batch['label'].to(device, non_blocking=_non_blocking)
            
            # AMP autocast
            with torch.amp.autocast('cuda', enabled=_use_amp):
                logits = model(embeddings, attention_mask)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def predict(
    model: nn.Module,
    dataloader,
    device: torch.device,
    config: TrainingConfig = None,
    use_amp: bool = None,
    non_blocking: bool = None,
    return_tensor: bool = False
):
    """
    테스트 데이터 예측 (AMP + Non-blocking Transfer)
    
    Args:
        model: 예측할 모델
        dataloader: 테스트 데이터로더
        device: 디바이스 (cuda/cpu)
        config: TrainingConfig 인스턴스 (우선 적용)
        use_amp: AMP 사용 여부 (config가 없을 때 사용)
        non_blocking: Non-blocking transfer 사용 여부 (config가 없을 때 사용)
        return_tensor: True면 torch.Tensor 반환, False면 numpy array 반환
    
    Returns:
        predictions: 예측 결과 (num_samples, num_classes)
            - return_tensor=True: torch.Tensor
            - return_tensor=False: numpy.ndarray
    """
    # 설정 결정
    if config is not None:
        _use_amp = config.use_amp
        _non_blocking = config.non_blocking
    else:
        _use_amp = use_amp if use_amp is not None else torch.cuda.is_available()
        _non_blocking = non_blocking if non_blocking is not None else True
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            # Non-blocking data transfer
            embeddings = batch['embedding'].to(device, non_blocking=_non_blocking)
            attention_mask = batch['attention_mask'].to(device, non_blocking=_non_blocking)
            
            # AMP autocast
            with torch.amp.autocast('cuda', enabled=_use_amp):
                logits = model(embeddings, attention_mask)
                probs = torch.sigmoid(logits)
            
            if return_tensor:
                all_predictions.append(probs.cpu())
            else:
                all_predictions.append(probs.cpu().numpy())
    
    if return_tensor:
        return torch.cat(all_predictions, dim=0)
    else:
        if np is None:
            raise ImportError("numpy is required for numpy array output. Install it or use return_tensor=True")
        return np.vstack(all_predictions)


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig = None,
    learning_rate: float = None,
    weight_decay: float = None,
    optimizer_type: str = 'adamw'
) -> torch.optim.Optimizer:
    """
    옵티마이저 생성 편의 함수
    
    Args:
        model: 모델
        config: TrainingConfig 인스턴스 (우선 적용)
        learning_rate: 학습률 (config가 없을 때 사용)
        weight_decay: L2 정규화 가중치 (config가 없을 때 사용)
        optimizer_type: 옵티마이저 종류 ('adamw', 'adam', 'sgd')
    
    Returns:
        optimizer: 옵티마이저 인스턴스
    """
    # 설정 결정
    if config is not None:
        _lr = config.learning_rate
        _wd = config.weight_decay
    else:
        _lr = learning_rate if learning_rate is not None else 1e-4
        _wd = weight_decay if weight_decay is not None else 1e-5
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=_lr,
            weight_decay=_wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=_lr,
            weight_decay=_wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=_lr,
            weight_decay=_wd,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig = None,
    factor: float = None,
    patience: int = None,
    min_lr: float = None,
    scheduler_type: str = 'reduce_on_plateau',
    t_0: int = None,
    t_mult: int = None
):
    """
    학습률 스케줄러 생성 편의 함수
    
    Args:
        optimizer: 옵티마이저
        config: TrainingConfig 인스턴스 (우선 적용)
        factor: LR 감소 비율 (config가 없을 때 사용)
        patience: 개선 없을 시 대기 에폭 수 (config가 없을 때 사용)
        min_lr: 최소 학습률 (config가 없을 때 사용)
        scheduler_type: 스케줄러 종류 ('reduce_on_plateau', 'cosine', 'step', 'cosine_warm_restarts')
        t_0: CosineAnnealingWarmRestarts의 첫 번째 restart까지 에폭 수 (기본값: 10)
        t_mult: CosineAnnealingWarmRestarts의 restart 주기 배수 (기본값: 2)
    
    Returns:
        scheduler: 스케줄러 인스턴스
    """
    # 설정 결정
    if config is not None:
        _factor = config.scheduler_factor
        _patience = config.scheduler_patience
        _min_lr = config.scheduler_min_lr
        _t_0 = getattr(config, 'cosine_t_0', None) or t_0 or 10
        _t_mult = getattr(config, 'cosine_t_mult', None) or t_mult or 2
    else:
        _factor = factor if factor is not None else 0.5
        _patience = patience if patience is not None else 3
        _min_lr = min_lr if min_lr is not None else 1e-7
        _t_0 = t_0 if t_0 is not None else 10
        _t_mult = t_mult if t_mult is not None else 2
    
    if scheduler_type.lower() == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=_factor,
            patience=_patience,
            min_lr=_min_lr
        )
    elif scheduler_type.lower() == 'cosine':
        # CosineAnnealingLR은 T_max가 필요함
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=_patience * 10,  # 대략적인 값
            eta_min=_min_lr
        )
    elif scheduler_type.lower() == 'cosine_warm_restarts':
        # CosineAnnealingWarmRestarts 추가
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=_t_0,
            T_mult=_t_mult,
            eta_min=_min_lr
        )
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=_patience,
            gamma=_factor
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_scaler(enabled: bool = None) -> torch.amp.GradScaler:
    """
    AMP GradScaler 생성 편의 함수
    
    Args:
        enabled: AMP 활성화 여부 (None이면 GPU 사용 여부에 따라 자동 결정)
    
    Returns:
        scaler: GradScaler 인스턴스
    """
    if enabled is None:
        enabled = torch.cuda.is_available()
    return torch.amp.GradScaler('cuda', enabled=enabled)


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
        """"
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

# =========================================================================
# IC 가중치 기반 손실 함수 (CAFA6 평가 지표 최적화)
# =========================================================================
# CAFA6 평가는 Information Content (IC) 가중치를 사용합니다.
# - 일반적인 GO term (Root): 가중치 ≈ 0 (맞춰도 점수 없음)
# - 특정적인 GO term (Leaf): 높은 가중치 (맞추면 높은 점수)
#
# ICWeightedBCELoss: 정답(Label=1)인 경우에 IC 가중치를 곱합니다.
# =========================================================================

from pathlib import Path


def load_ia_weights(
    ia_path: str,
    go_to_idx: Dict[str, int],
    num_classes: int,
    default_weight: float = 0.0
) -> torch.Tensor:
    """
    IA.tsv 파일에서 IC 가중치를 로드하고 GO term 인덱스에 맞게 매핑
    
    Args:
        ia_path: IA.tsv 파일 경로
        go_to_idx: GO term -> index 매핑 딕셔너리
        num_classes: 총 GO term 클래스 수
        default_weight: IA.tsv에 없는 GO term의 기본 가중치
    
    Returns:
        ic_weights: (num_classes,) 크기의 IC 가중치 텐서
    """
    import pandas as pd
    
    # IA.tsv 로드 (탭 구분, 헤더 없음)
    ia_df = pd.read_csv(ia_path, sep='\t', header=None, names=['GO_Term', 'IA_Weight'])
    
    # GO term -> IC 가중치 딕셔너리
    ia_dict = dict(zip(ia_df['GO_Term'], ia_df['IA_Weight']))
    
    # 인덱스 순서로 IC 가중치 배열 생성
    ic_weights = np.full(num_classes, default_weight, dtype=np.float32)
    
    matched_count = 0
    for go_term, idx in go_to_idx.items():
        if go_term in ia_dict:
            ic_weights[idx] = ia_dict[go_term]
            matched_count += 1
    
    print(f"✅ IC 가중치 로드 완료:")
    print(f"   IA.tsv 항목 수: {len(ia_df):,}")
    print(f"   매칭된 GO term 수: {matched_count:,} / {num_classes:,}")
    print(f"   IC 가중치 통계: min={ic_weights.min():.4f}, max={ic_weights.max():.4f}, mean={ic_weights.mean():.4f}")
    print(f"   0이 아닌 가중치 수: {(ic_weights > 0).sum():,}")
    
    return torch.from_numpy(ic_weights)


class ICWeightedBCELoss(nn.Module):
    """
    Information Content (IC) 가중치 기반 BCE Loss
    
    CAFA6 평가 지표에 맞게 설계된 손실 함수입니다.
    - 정답(Label=1)인 경우에 IC 가중치를 곱합니다.
    - 중요한(희귀한) GO term을 더 잘 맞추도록 학습합니다.
    
    수식:
        L = -1/N * Σ [ w_i * y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
        
        where:
            w_i: IC 가중치 (IA.tsv에서 로드)
            y_i: 실제 라벨 (0 또는 1)
            p_i: 예측 확률
    
    Args:
        ic_weights: (num_classes,) 크기의 IC 가중치 텐서
        normalize: IC 가중치 정규화 여부 (기본: True)
        pos_weight_scale: Positive 샘플 가중치 스케일 (기본: 1.0)
        reduction: 'mean', 'sum', 'none' 중 선택 (기본: 'mean')
    
    Usage:
        # IC 가중치 로드
        ic_weights = load_ia_weights('IA.tsv', go_to_idx, num_classes)
        
        # 손실 함수 생성
        criterion = ICWeightedBCELoss(ic_weights)
        
        # 학습
        loss = criterion(logits, labels)
    """
    
    def __init__(
        self,
        ic_weights: torch.Tensor,
        normalize: bool = True,
        pos_weight_scale: float = 1.0,
        reduction: str = 'mean'
    ):
        super(ICWeightedBCELoss, self).__init__()
        
        self.reduction = reduction
        self.pos_weight_scale = pos_weight_scale
        
        # IC 가중치 정규화 (선택적)
        if normalize:
            # 0이 아닌 가중치만 사용하여 정규화
            nonzero_mask = ic_weights > 0
            if nonzero_mask.sum() > 0:
                nonzero_weights = ic_weights[nonzero_mask]
                # mean=1이 되도록 정규화
                ic_weights = ic_weights / (nonzero_weights.mean() + 1e-8)
        
        # 최소 가중치 설정 (0인 경우에도 최소한의 학습)
        ic_weights = torch.clamp(ic_weights, min=0.1)
        
        # 버퍼로 등록 (모델 저장/로드 시 함께 저장됨, 학습되지 않음)
        self.register_buffer('ic_weights', ic_weights)
        
        print(f"✅ ICWeightedBCELoss 초기화:")
        print(f"   정규화: {'✅' if normalize else '❌'}")
        print(f"   가중치 범위: [{self.ic_weights.min():.4f}, {self.ic_weights.max():.4f}]")
        print(f"   pos_weight_scale: {pos_weight_scale}")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - 모델 출력 (sigmoid 전)
            labels: (batch, num_classes) - 실제 라벨 (0 또는 1)
        
        Returns:
            loss: 스칼라 손실 값
        """
        # Sigmoid 적용
        probs = torch.sigmoid(logits)
        
        # 수치 안정성을 위한 클램핑
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # Binary Cross Entropy 계산
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        bce_pos = -labels * torch.log(probs)           # Positive: -y * log(p)
        bce_neg = -(1 - labels) * torch.log(1 - probs)  # Negative: -(1-y) * log(1-p)
        
        # IC 가중치 적용 (Positive만)
        # Positive 샘플에만 IC 가중치를 곱함
        weighted_bce_pos = bce_pos * self.ic_weights.unsqueeze(0) * self.pos_weight_scale
        
        # 총 손실
        loss = weighted_bce_pos + bce_neg
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ICWeightedBCEWithLogitsLoss(nn.Module):
    """
    Information Content (IC) 가중치 기반 BCE Loss (LogSumExp 안정 버전)
    
    BCEWithLogitsLoss와 같이 수치적으로 안정적인 구현입니다.
    
    Args:
        ic_weights: (num_classes,) 크기의 IC 가중치 텐서
        normalize: IC 가중치 정규화 여부 (기본: True)
        pos_weight_scale: Positive 샘플 가중치 스케일 (기본: 1.0)
        neg_weight: Negative 샘플 가중치 (기본: 1.0)
        reduction: 'mean', 'sum', 'none' 중 선택 (기본: 'mean')
    """
    
    def __init__(
        self,
        ic_weights: torch.Tensor,
        normalize: bool = True,
        pos_weight_scale: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super(ICWeightedBCEWithLogitsLoss, self).__init__()
        
        self.reduction = reduction
        self.pos_weight_scale = pos_weight_scale
        self.neg_weight = neg_weight
        
        # IC 가중치 정규화
        if normalize:
            nonzero_mask = ic_weights > 0
            if nonzero_mask.sum() > 0:
                nonzero_weights = ic_weights[nonzero_mask]
                ic_weights = ic_weights / (nonzero_weights.mean() + 1e-8)
        
        # 최소 가중치 설정
        ic_weights = torch.clamp(ic_weights, min=0.1)
        
        self.register_buffer('ic_weights', ic_weights)
        
        print(f"✅ ICWeightedBCEWithLogitsLoss 초기화:")
        print(f"   정규화: {'✅' if normalize else '❌'}")
        print(f"   가중치 범위: [{self.ic_weights.min():.4f}, {self.ic_weights.max():.4f}]")
        print(f"   pos_weight_scale: {pos_weight_scale}, neg_weight: {neg_weight}")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        수치적으로 안정적인 Weighted BCE 계산
        
        BCEWithLogits 공식 (수치 안정성):
            max(x, 0) - x * y + log(1 + exp(-|x|))
        """
        # 가중치 계산: positive는 IC 가중치, negative는 고정 가중치
        # weight = labels * ic_weights * pos_scale + (1 - labels) * neg_weight
        pos_weight = self.ic_weights.unsqueeze(0) * self.pos_weight_scale  # (1, num_classes)
        
        # 수치 안정적인 BCE 계산
        # loss = max(logits, 0) - logits * labels + log(1 + exp(-|logits|))
        max_val = torch.clamp(logits, min=0)
        bce = max_val - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))
        
        # Positive/Negative 분리하여 가중치 적용
        # Positive loss: labels * bce * pos_weight
        # Negative loss: (1 - labels) * bce * neg_weight
        weighted_loss = labels * bce * pos_weight + (1 - labels) * bce * self.neg_weight
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


# =========================================================================
# Weighted F-max 및 S-min 계산 함수 (CAFA6 평가 지표)
# =========================================================================

def calculate_weighted_fmax(
    preds: np.ndarray,
    labels: np.ndarray,
    ia_weights: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    Weighted F-max 계산 (CAFA6 공식 평가 지표)
    
    Args:
        preds: (N_proteins, N_terms) 예측 확률 (0~1)
        labels: (N_proteins, N_terms) 실제 라벨 (0 또는 1)
        ia_weights: (N_terms,) IC 가중치
        thresholds: 스캔할 임계값 배열 (기본: 0~1, 101개)
    
    Returns:
        best_fmax: 최고 F-max 점수
        best_threshold: 최적 임계값
        all_f1_scores: 모든 임계값에서의 F1 점수 리스트
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    # Ground Truth의 총 IC (Recall 분모)
    total_true_ic = labels @ ia_weights  # (N_proteins,)
    total_true_ic = np.maximum(total_true_ic, 1e-10)  # Division by zero 방지
    
    best_fmax = 0.0
    best_threshold = 0.0
    all_f1_scores = []
    
    for t in thresholds:
        # 1. 임계값 적용
        pred_binary = (preds >= t).astype(np.float32)
        
        # 2. True Positive (교집합)
        tp_mask = pred_binary * labels
        weighted_tp = tp_mask @ ia_weights  # (N_proteins,)
        
        # 3. Prediction의 총 IC (Precision 분모)
        weighted_pred = pred_binary @ ia_weights  # (N_proteins,)
        
        # 4. Weighted Precision & Recall
        precision = np.divide(
            weighted_tp, weighted_pred,
            out=np.zeros_like(weighted_tp),
            where=weighted_pred != 0
        )
        recall = weighted_tp / total_true_ic
        
        # 5. 평균
        avg_prec = np.mean(precision)
        avg_rec = np.mean(recall)
        
        # 6. F1 계산
        if (avg_prec + avg_rec) > 0:
            f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
        else:
            f1 = 0.0
        
        all_f1_scores.append(f1)
        
        # 7. 최고 F-max 업데이트
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = t
    
    return best_fmax, best_threshold, all_f1_scores


def calculate_smin(
    preds: np.ndarray,
    labels: np.ndarray,
    ia_weights: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    S-min (Semantic Distance) 계산
    
    Args:
        preds: (N_proteins, N_terms) 예측 확률 (0~1)
        labels: (N_proteins, N_terms) 실제 라벨 (0 또는 1)
        ia_weights: (N_terms,) IC 가중치
        thresholds: 스캔할 임계값 배열
    
    Returns:
        min_s: 최소 S-score
        best_threshold: 최적 임계값
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    min_s = float('inf')
    best_threshold = 0.0
    
    for t in thresholds:
        pred_binary = (preds >= t).astype(np.float32)
        
        # False Negatives (RU: Remaining Uncertainty)
        fn_mask = labels * (1 - pred_binary)
        ru_per_protein = fn_mask @ ia_weights
        avg_ru = np.mean(ru_per_protein)
        
        # False Positives (MI: Misinformation)
        fp_mask = pred_binary * (1 - labels)
        mi_per_protein = fp_mask @ ia_weights
        avg_mi = np.mean(mi_per_protein)
        
        # Euclidean Distance
        s_score = np.sqrt(avg_ru**2 + avg_mi**2)
        
        if s_score < min_s:
            min_s = s_score
            best_threshold = t
    
    return min_s, best_threshold


def evaluate_cafa6_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    ia_weights: np.ndarray,
    thresholds: np.ndarray = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    CAFA6 전체 평가 지표 계산
    
    Args:
        preds: 예측 확률
        labels: 실제 라벨
        ia_weights: IC 가중치
        thresholds: 임계값 배열
        verbose: 결과 출력 여부
    
    Returns:
        dict: {
            'fmax': float,
            'fmax_threshold': float,
            'smin': float,
            'smin_threshold': float,
            'f1_scores': list
        }
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    # Weighted F-max
    fmax, fmax_t, f1_scores = calculate_weighted_fmax(preds, labels, ia_weights, thresholds)
    
    # S-min
    smin, smin_t = calculate_smin(preds, labels, ia_weights, thresholds)
    
    results = {
        'fmax': fmax,
        'fmax_threshold': fmax_t,
        'smin': smin,
        'smin_threshold': smin_t,
        'f1_scores': f1_scores
    }
    
    if verbose:
        print("=" * 60)
        print("📊 CAFA6 평가 결과")
        print("=" * 60)
        print(f"  Weighted F-max: {fmax:.4f} (threshold={fmax_t:.2f})")
        print(f"  S-min: {smin:.4f} (threshold={smin_t:.2f})")
        print("=" * 60)
    
    return results


# =========================================================================
# 손실 함수 생성 편의 함수
# =========================================================================

def create_ic_weighted_criterion(
    ia_path: str,
    go_to_idx: Dict[str, int],
    num_classes: int,
    normalize: bool = True,
    pos_weight_scale: float = 1.0,
    neg_weight: float = 1.0,
    device: str = None
) -> ICWeightedBCEWithLogitsLoss:
    """
    IC 가중치 기반 손실 함수 생성 편의 함수
    
    Args:
        ia_path: IA.tsv 파일 경로
        go_to_idx: GO term -> index 매핑
        num_classes: GO term 클래스 수
        normalize: 가중치 정규화 여부
        pos_weight_scale: Positive 가중치 스케일
        neg_weight: Negative 가중치
        device: 디바이스
    
    Returns:
        ICWeightedBCEWithLogitsLoss 인스턴스
    """
    # IC 가중치 로드
    ic_weights = load_ia_weights(ia_path, go_to_idx, num_classes)
    
    # 손실 함수 생성
    criterion = ICWeightedBCEWithLogitsLoss(
        ic_weights=ic_weights,
        normalize=normalize,
        pos_weight_scale=pos_weight_scale,
        neg_weight=neg_weight
    )
    
    # 디바이스로 이동
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = criterion.to(device)
    
    return criterion


# =========================================================================
# 메인 (테스트용)
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Model Module Test")
    print("=" * 70)
    
    # 설정 테스트
    config = ModelConfig(
        embedding_dim=1280,
        num_classes=1000,  # 테스트용 작은 값
        conv_channels=[256, 512, 512],
        kernel_sizes=[7, 5, 3],
        fc_dims=[512],
        dropout=0.3,
        use_residual=True
    )
    print(config.summary())
    
    # 모델 생성 테스트
    model = Hierarchical1DCNNClassifier.from_config(config)
    print(model.summary())
    
    # Forward pass 테스트
    test_model(model, seq_length=100, batch_size=2)
    
    # IC 가중치 손실 함수 테스트
    print("\n" + "=" * 70)
    print("🧪 ICWeightedBCELoss Test")
    print("=" * 70)
    
    # 더미 IC 가중치 생성
    dummy_ic_weights = torch.rand(1000)
    dummy_ic_weights[::10] = 0  # 일부 0으로 설정
    
    criterion = ICWeightedBCEWithLogitsLoss(dummy_ic_weights, normalize=True)
    
    # 더미 데이터로 테스트
    dummy_logits = torch.randn(4, 1000)
    dummy_labels = torch.zeros(4, 1000)
    dummy_labels[:, :50] = 1  # 일부 positive
    
    loss = criterion(dummy_logits, dummy_labels)
    print(f"  Loss: {loss.item():.4f}")
    print("  ✅ ICWeightedBCELoss 테스트 성공!")

