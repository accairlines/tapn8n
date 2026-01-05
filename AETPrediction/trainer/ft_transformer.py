"""
FT-Transformer (Feature Tokenizer Transformer) implementation for tabular data
Based on the paper: "Revisiting Deep Learning Models for Tabular Data"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class Tokenizer(nn.Module):
    """Feature Tokenizer for numerical and categorical features"""
    
    def __init__(self, num_numerical, num_categories, d_token):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categories = num_categories
        self.d_token = d_token
        
        # Numerical feature tokenizer
        if num_numerical > 0:
            self.numerical_tokenizer = nn.Linear(1, d_token)
        
        # Categorical feature tokenizer (embedding)
        if num_categories > 0:
            # We'll use a large embedding size and project down
            # For simplicity, we'll use a single embedding table with max categories
            # In practice, each categorical feature should have its own embedding
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(1000000, d_token)  # Large vocab size for hash-based encoding
                for _ in range(num_categories)
            ])
    
    def forward(self, x_numerical, x_categorical):
        """
        Args:
            x_numerical: (batch_size, num_numerical)
            x_categorical: (batch_size, num_categories)
        Returns:
            tokens: (batch_size, num_features, d_token)
        """
        tokens = []
        
        # Process numerical features
        if self.num_numerical > 0 and x_numerical is not None:
            # Reshape to (batch_size, num_numerical, 1)
            x_num = x_numerical.unsqueeze(-1)
            num_tokens = self.numerical_tokenizer(x_num)  # (batch_size, num_numerical, d_token)
            tokens.append(num_tokens)
        
        # Process categorical features
        if self.num_categories > 0 and x_categorical is not None:
            cat_tokens = []
            for i, emb in enumerate(self.category_embeddings):
                # Clamp categorical values to valid range
                cat_values = x_categorical[:, i].long()
                cat_values = torch.clamp(cat_values, 0, emb.num_embeddings - 1)
                cat_token = emb(cat_values).unsqueeze(1)  # (batch_size, 1, d_token)
                cat_tokens.append(cat_token)
            if cat_tokens:
                cat_tokens = torch.cat(cat_tokens, dim=1)  # (batch_size, num_categories, d_token)
                tokens.append(cat_tokens)
        
        if tokens:
            return torch.cat(tokens, dim=1)  # (batch_size, num_features, d_token)
        else:
            raise ValueError("No features provided")


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    
    def __init__(self, d_token, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer model for regression"""
    
    def __init__(
        self,
        num_numerical,
        num_categories,
        d_token=192,
        n_layers=3,
        n_heads=8,
        d_ff=768,
        dropout=0.1
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categories = num_categories
        self.d_token = d_token
        
        # Feature tokenizer
        self.tokenizer = Tokenizer(num_numerical, num_categories, d_token)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x_numerical, x_categorical):
        """
        Args:
            x_numerical: (batch_size, num_numerical)
            x_categorical: (batch_size, num_categories)
        Returns:
            output: (batch_size, 1)
        """
        # Tokenize features
        tokens = self.tokenizer(x_numerical, x_categorical)  # (batch_size, num_features, d_token)
        
        # Add CLS token
        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)  # (batch_size, num_features + 1, d_token)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Extract CLS token and predict
        cls_output = x[:, 0, :]  # (batch_size, d_token)
        output = self.head(self.norm(cls_output))  # (batch_size, 1)
        
        return output


class FTTransformerTrainer:
    """Trainer class for FT-Transformer"""
    
    def __init__(
        self,
        num_numerical,
        num_categories,
        d_token=192,
        n_layers=3,
        n_heads=8,
        d_ff=768,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=256,
        n_epochs=100,
        device=None
    ):
        self.num_numerical = num_numerical
        self.num_categories = num_categories
        self.d_token = d_token
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = FTTransformer(
            num_numerical=num_numerical,
            num_categories=num_categories,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        # Scaler for numerical features
        self.numerical_scaler = StandardScaler()
        
    def train(self, X_numerical, X_categorical, y):
        """Train the model"""
        logger.info("Training FT-Transformer...")
        
        # Convert to numpy if needed
        if isinstance(X_numerical, pd.DataFrame):
            X_numerical = X_numerical.values
        if isinstance(X_categorical, pd.DataFrame):
            X_categorical = X_categorical.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.flatten()
        
        # Scale numerical features
        if self.num_numerical > 0:
            X_numerical = self.numerical_scaler.fit_transform(X_numerical)
        
        # Convert to tensors
        if self.num_numerical > 0:
            X_num_tensor = torch.FloatTensor(X_numerical).to(self.device)
        else:
            X_num_tensor = None
        
        if self.num_categories > 0:
            X_cat_tensor = torch.LongTensor(X_categorical).to(self.device)
        else:
            X_cat_tensor = None
        
        y_tensor = torch.FloatTensor(y).to(self.device).unsqueeze(1)
        
        # Training loop
        self.model.train()
        n_samples = len(y)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = torch.randperm(n_samples).to(self.device)
            if X_num_tensor is not None:
                X_num_shuffled = X_num_tensor[indices]
            else:
                X_num_shuffled = None
            if X_cat_tensor is not None:
                X_cat_shuffled = X_cat_tensor[indices]
            else:
                X_cat_shuffled = None
            y_shuffled = y_tensor[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                batch_X_num = X_num_shuffled[start_idx:end_idx] if X_num_shuffled is not None else None
                batch_X_cat = X_cat_shuffled[start_idx:end_idx] if X_cat_shuffled is not None else None
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_X_num, batch_X_cat)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Training completed. Best loss: {best_loss:.4f}")
        return self.model, self.numerical_scaler
    
    def predict(self, X_numerical, X_categorical):
        """Make predictions"""
        self.model.eval()
        
        # Convert to numpy if needed
        if isinstance(X_numerical, pd.DataFrame):
            X_numerical = X_numerical.values
        if isinstance(X_categorical, pd.DataFrame):
            X_categorical = X_categorical.values
        
        # Scale numerical features
        if self.num_numerical > 0:
            X_numerical = self.numerical_scaler.transform(X_numerical)
        
        # Convert to tensors
        if self.num_numerical > 0:
            X_num_tensor = torch.FloatTensor(X_numerical).to(self.device)
        else:
            X_num_tensor = None
        
        if self.num_categories > 0:
            X_cat_tensor = torch.LongTensor(X_categorical).to(self.device)
        else:
            X_cat_tensor = None
        
        with torch.no_grad():
            predictions = self.model(X_num_tensor, X_cat_tensor)
        
        return predictions.cpu().numpy().flatten()

