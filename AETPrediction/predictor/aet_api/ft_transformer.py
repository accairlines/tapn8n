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
            # Reduced embedding size to save memory
            # Using smaller vocab size (10000 instead of 1000000)
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(10000, d_token)  # Reduced vocab size to save memory
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
            result = torch.cat(tokens, dim=1)  # (batch_size, num_features, d_token)
            if result.size(1) == 0:
                raise ValueError(
                    f"Tokenized sequence has 0 length. "
                    f"num_numerical={self.num_numerical}, num_categories={self.num_categories}"
                )
            return result
        else:
            raise ValueError("No features provided")


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    
    def __init__(self, d_token, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Validate that d_token is divisible by n_heads
        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_heads ({n_heads}). "
                f"Current division: {d_token / n_heads}"
            )
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
        # Validate input shape
        if x.size(1) == 0:
            raise ValueError(f"Sequence length is 0. Input shape: {x.shape}")
        
        normalized = self.norm1(x)
        attn_output, _ = self.attention(normalized, normalized, normalized)
        x = x + attn_output
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
        logger.debug(f"Forward pass - x_numerical: {x_numerical.shape if x_numerical is not None else None}, "
                    f"x_categorical: {x_categorical.shape if x_categorical is not None else None}")
        
        # Tokenize features
        try:
            tokens = self.tokenizer(x_numerical, x_categorical)  # (batch_size, num_features, d_token)
            logger.debug(f"Tokenized features shape: {tokens.shape}")
        except Exception as e:
            logger.error(f"Error in tokenizer: {str(e)}")
            raise
        
        # Validate tokenized output
        if tokens.size(1) == 0:
            raise ValueError(
                f"Tokenized features have 0 length. "
                f"x_numerical shape: {x_numerical.shape if x_numerical is not None else None}, "
                f"x_categorical shape: {x_categorical.shape if x_categorical is not None else None}"
            )
        
        # Add CLS token
        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)  # (batch_size, num_features + 1, d_token)
        logger.debug(f"After adding CLS token, shape: {x.shape}")
        
        # Apply transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            try:
                logger.debug(f"Applying transformer block {i + 1}/{len(self.transformer_blocks)}")
                x = block(x)
                logger.debug(f"After block {i + 1}, shape: {x.shape}")
            except Exception as e:
                logger.error(f"Error in transformer block {i + 1}: {str(e)}")
                logger.error(f"  Input shape to block: {x.shape}")
                raise
        
        # Extract CLS token and predict
        cls_output = x[:, 0, :]  # (batch_size, d_token)
        logger.debug(f"CLS token output shape: {cls_output.shape}")
        output = self.head(self.norm(cls_output))  # (batch_size, 1)
        logger.debug(f"Final output shape: {output.shape}")
        
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
        logger.info(f"Initializing FT-Transformer model with {num_numerical} numerical and {num_categories} categorical features...")
        self.model = FTTransformer(
            num_numerical=num_numerical,
            num_categories=num_categories,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        ).to(self.device)
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized. Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        # Scaler for numerical features
        self.numerical_scaler = StandardScaler()
    
    def train(self, X_numerical, X_categorical, y):
        """Train the model"""
        logger.info("Training FT-Transformer...")
        logger.info(f"Model configuration: d_token={self.d_token}, n_layers={self.n_layers}, "
                   f"n_heads={self.n_heads}, d_ff={self.d_ff}, batch_size={self.batch_size}, "
                   f"n_epochs={self.n_epochs}")
        
        # Convert to numpy if needed
        logger.info("Converting input data to numpy arrays...")
        if isinstance(X_numerical, pd.DataFrame):
            X_numerical = X_numerical.values
        if isinstance(X_categorical, pd.DataFrame):
            X_categorical = X_categorical.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.flatten()
        
        logger.info(f"Input shapes - X_numerical: {X_numerical.shape if X_numerical is not None else None}, "
                   f"X_categorical: {X_categorical.shape if X_categorical is not None else None}, "
                   f"y: {y.shape}")
        
        # Scale numerical features
        if self.num_numerical > 0:
            logger.info("Scaling numerical features...")
            X_numerical = self.numerical_scaler.fit_transform(X_numerical)
            logger.info("Numerical features scaled successfully")
        
        # Keep data on CPU to save memory, only move batches to device
        logger.info("Converting data to tensors...")
        if self.num_numerical > 0:
            X_num_tensor = torch.FloatTensor(X_numerical)  # Keep on CPU
            logger.info(f"Created numerical tensor with shape: {X_num_tensor.shape}")
        else:
            X_num_tensor = None
            logger.info("No numerical features to convert")
        
        if self.num_categories > 0:
            X_cat_tensor = torch.LongTensor(X_categorical)  # Keep on CPU
            logger.info(f"Created categorical tensor with shape: {X_cat_tensor.shape}")
        else:
            X_cat_tensor = None
            logger.info("No categorical features to convert")
        
        y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Keep on CPU
        logger.info(f"Created target tensor with shape: {y_tensor.shape}")
        
        # Training loop
        logger.info("Starting training loop...")
        self.model.train()
        n_samples = len(y)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Training setup: {n_samples} samples, {n_batches} batches per epoch")
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.n_epochs}")
            epoch_loss = 0.0
            
            # Shuffle data indices on CPU to save memory
            indices = torch.randperm(n_samples)
            
            for batch_idx in range(n_batches):
                try:
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    logger.info(f"  Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx-1})")
                    
                    # Only move batch to device, not entire dataset
                    if X_num_tensor is not None:
                        batch_X_num = X_num_tensor[batch_indices].to(self.device)
                        logger.debug(f"    Batch numerical shape: {batch_X_num.shape}")
                    else:
                        batch_X_num = None
                    if X_cat_tensor is not None:
                        batch_X_cat = X_cat_tensor[batch_indices].to(self.device)
                        logger.debug(f"    Batch categorical shape: {batch_X_cat.shape}")
                    else:
                        batch_X_cat = None
                    batch_y = y_tensor[batch_indices].to(self.device)
                    logger.debug(f"    Batch target shape: {batch_y.shape}")
                    
                    # Forward pass
                    logger.debug(f"    Starting forward pass...")
                    self.optimizer.zero_grad()
                    predictions = self.model(batch_X_num, batch_X_cat)
                    logger.debug(f"    Predictions shape: {predictions.shape}")
                    loss = self.criterion(predictions, batch_y)
                    logger.debug(f"    Batch loss: {loss.item():.4f}")
                    
                    # Backward pass
                    logger.debug(f"    Starting backward pass...")
                    loss.backward()
                    self.optimizer.step()
                    logger.debug(f"    Backward pass completed")
                    
                    epoch_loss += loss.item()
                    
                    # Clear batch tensors to free memory
                    del batch_X_num, batch_X_cat, batch_y, predictions, loss
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error in epoch {epoch + 1}, batch {batch_idx + 1}/{n_batches}")
                    logger.error(f"  Batch indices: {start_idx}-{end_idx-1}")
                    logger.error(f"  Batch numerical shape: {batch_X_num.shape if batch_X_num is not None else None}")
                    logger.error(f"  Batch categorical shape: {batch_X_cat.shape if batch_X_cat is not None else None}")
                    logger.error(f"  Error: {str(e)}", exc_info=True)
                    raise
            
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch + 1}/{self.n_epochs} completed. Average loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                logger.info(f"  New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  Loss did not improve. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
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
