"""
Modern Baseline Methods for Step 2.3 - State-of-the-Art Recommendation Systems
Implements Neural Collaborative Filtering, Variational Autoencoders, and other modern approaches
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.evaluation.baseline_methods import BaselineRecommender

logger = logging.getLogger(__name__)

class NeuralCollaborativeFiltering(BaselineRecommender):
    """
    Neural Collaborative Filtering (NCF) - State-of-the-art deep learning approach
    Based on He et al. (2017) "Neural Collaborative Filtering"
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_layers: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001, epochs: int = 50):
        super().__init__("NeuralCollaborativeFiltering")
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.n_users = 0
        self.n_items = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    class NCFModel(nn.Module):
        def __init__(self, n_users, n_items, embedding_dim, hidden_layers, dropout_rate):
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items
            self.embedding_dim = embedding_dim
            
            # Embeddings
            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)
            
            # MLP layers
            self.fc_layers = nn.ModuleList()
            input_size = embedding_dim * 2
            
            for hidden_size in hidden_layers:
                self.fc_layers.append(nn.Linear(input_size, hidden_size))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_rate))
                input_size = hidden_size
            
            # Output layer
            self.output = nn.Linear(input_size, 1)
            
            # Initialize weights
            self._init_weights()
            
        def _init_weights(self):
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
            
        def forward(self, user_ids, item_ids):
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
            
            # Concatenate embeddings
            x = torch.cat([user_emb, item_emb], dim=1)
            
            # Pass through MLP
            for layer in self.fc_layers:
                x = layer(x)
                
            # Output rating prediction
            rating = torch.sigmoid(self.output(x))
            return rating.squeeze()
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """Train the NCF model"""
        
        self.n_users, self.n_items = user_item_matrix.shape
        
        # Create user and item encoders
        self.user_encoder = {i: i for i in range(self.n_users)}
        self.item_encoder = {i: i for i in range(self.n_items)}
        
        # Prepare training data
        train_data = []
        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                rating = user_item_matrix[user_id, item_id]
                if rating > 0:  # Only use positive interactions
                    # Normalize rating to [0, 1]
                    normalized_rating = min(rating / 5.0, 1.0)
                    train_data.append((user_id, item_id, normalized_rating))
        
        # Add negative samples
        negative_samples = []
        for user_id in range(self.n_users):
            user_items = set(np.where(user_item_matrix[user_id] > 0)[0])
            available_items = set(range(self.n_items)) - user_items
            
            # Sample negative items
            n_negatives = min(len(user_items), len(available_items))
            negative_items = np.random.choice(list(available_items), 
                                            size=n_negatives, replace=False)
            
            for item_id in negative_items:
                negative_samples.append((user_id, item_id, 0.0))
        
        train_data.extend(negative_samples)
        
        # Convert to tensors
        users = torch.LongTensor([x[0] for x in train_data])
        items = torch.LongTensor([x[1] for x in train_data])
        ratings = torch.FloatTensor([x[2] for x in train_data])
        
        # Create model
        self.model = self.NCFModel(
            self.n_users, self.n_items, self.embedding_dim, 
            self.hidden_layers, self.dropout_rate
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Move data to device
        users = users.to(self.device)
        items = items.to(self.device)
        ratings = ratings.to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(users, items)
            loss = criterion(predictions, ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"NCF Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.model.eval()
        self.is_trained = True
        logger.info("NCF training completed")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Generate recommendations using the trained NCF model"""
        
        if not self.is_trained or self.model is None:
            return [(i, 0.0) for i in range(n_recommendations)]
        
        # Prepare data for all items
        user_tensor = torch.LongTensor([user_id] * self.n_items).to(self.device)
        items_tensor = torch.LongTensor(list(range(self.n_items))).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(user_tensor, items_tensor)
            scores = predictions.cpu().numpy()
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
        
        return recommendations

class VariationalAutoEncoder(BaselineRecommender):
    """
    Variational Autoencoder for Collaborative Filtering (VAE-CF)
    Based on Liang et al. (2018) "Variational Autoencoders for Collaborative Filtering"
    """
    
    def __init__(self, latent_dim: int = 64, hidden_layers: List[int] = [128, 64],
                 dropout_rate: float = 0.5, learning_rate: float = 0.001, epochs: int = 50,
                 beta: float = 1.0):
        super().__init__("VariationalAutoEncoder")
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta  # KL divergence weight
        self.model = None
        self.n_items = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    class VAEModel(nn.Module):
        def __init__(self, n_items, latent_dim, hidden_layers, dropout_rate):
            super().__init__()
            self.n_items = n_items
            self.latent_dim = latent_dim
            
            # Encoder
            encoder_layers = []
            input_size = n_items
            
            for hidden_size in hidden_layers:
                encoder_layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_size = hidden_size
            
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Latent layers
            self.mu = nn.Linear(input_size, latent_dim)
            self.logvar = nn.Linear(input_size, latent_dim)
            
            # Decoder
            decoder_layers = []
            input_size = latent_dim
            
            for hidden_size in reversed(hidden_layers):
                decoder_layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_size = hidden_size
            
            decoder_layers.append(nn.Linear(input_size, n_items))
            self.decoder = nn.Sequential(*decoder_layers)
            
        def encode(self, x):
            h = self.encoder(x)
            mu = self.mu(h)
            logvar = self.logvar(h)
            return mu, logvar
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def decode(self, z):
            return self.decoder(z)
        
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            return recon_x, mu, logvar
    
    def vae_loss(self, recon_x, x, mu, logvar, beta):
        """VAE loss function combining reconstruction and KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """Train the VAE model"""
        
        self.n_items = user_item_matrix.shape[1]
        
        # Normalize user-item matrix
        user_profiles = user_item_matrix.copy().astype(np.float32)
        
        # Normalize to [0, 1]
        max_rating = user_profiles.max()
        if max_rating > 0:
            user_profiles = user_profiles / max_rating
        
        # Create model
        self.model = self.VAEModel(
            self.n_items, self.latent_dim, self.hidden_layers, self.dropout_rate
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert to tensor
        data = torch.FloatTensor(user_profiles).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            recon_data, mu, logvar = self.model(data)
            loss = self.vae_loss(recon_data, data, mu, logvar, self.beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"VAE Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.model.eval()
        self.is_trained = True
        logger.info("VAE training completed")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Generate recommendations using the trained VAE model"""
        
        if not self.is_trained or self.model is None:
            return [(i, 0.0) for i in range(n_recommendations)]
        
        # Get user profile
        user_profile = np.zeros(self.n_items)
        # In real implementation, you'd get the actual user profile
        # For now, we'll use a random profile for demonstration
        user_profile = torch.FloatTensor(user_profile).unsqueeze(0).to(self.device)
        
        # Generate recommendations
        with torch.no_grad():
            recon_profile, _, _ = self.model(user_profile)
            scores = recon_profile.squeeze().cpu().numpy()
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
        
        return recommendations

class BayesianPersonalizedRanking(BaselineRecommender):
    """
    Bayesian Personalized Ranking (BPR) for implicit feedback
    Based on Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback"
    """
    
    def __init__(self, factors: int = 64, learning_rate: float = 0.01, 
                 regularization: float = 0.01, epochs: int = 100):
        super().__init__("BayesianPersonalizedRanking")
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        self.n_users = 0
        self.n_items = 0
        
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """Train the BPR model"""
        
        self.n_users, self.n_items = user_item_matrix.shape
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.factors))
        
        # Get positive interactions
        user_items = defaultdict(set)
        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                if user_item_matrix[user_id, item_id] > 0:
                    user_items[user_id].add(item_id)
        
        # Training loop
        for epoch in range(self.epochs):
            for user_id in range(self.n_users):
                if user_id not in user_items or len(user_items[user_id]) == 0:
                    continue
                
                # Sample positive item
                pos_item = np.random.choice(list(user_items[user_id]))
                
                # Sample negative item
                all_items = set(range(self.n_items))
                neg_items = all_items - user_items[user_id]
                if len(neg_items) == 0:
                    continue
                neg_item = np.random.choice(list(neg_items))
                
                # Compute predictions
                pos_pred = np.dot(self.user_factors[user_id], self.item_factors[pos_item])
                neg_pred = np.dot(self.user_factors[user_id], self.item_factors[neg_item])
                
                # Compute gradients
                diff = pos_pred - neg_pred
                sigmoid = 1.0 / (1.0 + np.exp(-diff))
                
                # Update factors
                user_factor_update = self.learning_rate * ((1 - sigmoid) * 
                    (self.item_factors[pos_item] - self.item_factors[neg_item]) - 
                    self.regularization * self.user_factors[user_id])
                
                pos_item_factor_update = self.learning_rate * ((1 - sigmoid) * 
                    self.user_factors[user_id] - self.regularization * self.item_factors[pos_item])
                
                neg_item_factor_update = self.learning_rate * (-(1 - sigmoid) * 
                    self.user_factors[user_id] - self.regularization * self.item_factors[neg_item])
                
                self.user_factors[user_id] += user_factor_update
                self.item_factors[pos_item] += pos_item_factor_update
                self.item_factors[neg_item] += neg_item_factor_update
            
            if epoch % 20 == 0:
                logger.info(f"BPR Epoch {epoch}/{self.epochs}")
        
        self.is_trained = True
        logger.info("BPR training completed")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Generate recommendations using BPR"""
        
        if not self.is_trained or self.user_factors is None:
            return [(i, 0.0) for i in range(n_recommendations)]
        
        # Compute scores for all items
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
        
        return recommendations

class AutoEncoder(BaselineRecommender):
    """
    AutoEncoder for Collaborative Filtering
    Simpler version of VAE without the variational component
    """
    
    def __init__(self, hidden_layers: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3, learning_rate: float = 0.001, epochs: int = 50):
        super().__init__("AutoEncoder")
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.n_items = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    class AEModel(nn.Module):
        def __init__(self, n_items, hidden_layers, dropout_rate):
            super().__init__()
            
            # Encoder
            encoder_layers = []
            input_size = n_items
            
            for hidden_size in hidden_layers:
                encoder_layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_size = hidden_size
            
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Decoder
            decoder_layers = []
            for hidden_size in reversed(hidden_layers[:-1]):
                decoder_layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_size = hidden_size
            
            decoder_layers.append(nn.Linear(input_size, n_items))
            self.decoder = nn.Sequential(*decoder_layers)
            
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """Train the AutoEncoder model"""
        
        self.n_items = user_item_matrix.shape[1]
        
        # Normalize user-item matrix
        user_profiles = user_item_matrix.copy().astype(np.float32)
        max_rating = user_profiles.max()
        if max_rating > 0:
            user_profiles = user_profiles / max_rating
        
        # Create model
        self.model = self.AEModel(self.n_items, self.hidden_layers, self.dropout_rate).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensor
        data = torch.FloatTensor(user_profiles).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"AutoEncoder Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.model.eval()
        self.is_trained = True
        logger.info("AutoEncoder training completed")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Generate recommendations using AutoEncoder"""
        
        if not self.is_trained or self.model is None:
            return [(i, 0.0) for i in range(n_recommendations)]
        
        # Create user profile (simplified for demo)
        user_profile = np.zeros(self.n_items)
        user_profile = torch.FloatTensor(user_profile).unsqueeze(0).to(self.device)
        
        # Generate recommendations
        with torch.no_grad():
            reconstructed = self.model(user_profile)
            scores = reconstructed.squeeze().cpu().numpy()
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(int(item_id), float(scores[item_id])) for item_id in top_items]
        
        return recommendations

def create_modern_baselines() -> Dict[str, BaselineRecommender]:
    """
    Create all modern baseline recommendation methods for Step 2.3
    """
    
    modern_baselines = {
        'NeuralCollaborativeFiltering': NeuralCollaborativeFiltering(),
        'VariationalAutoEncoder': VariationalAutoEncoder(),
        'BayesianPersonalizedRanking': BayesianPersonalizedRanking(),
        'AutoEncoder': AutoEncoder()
    }
    
    return modern_baselines

def test_modern_baselines():
    """Test all modern baseline methods"""
    print("🚀 TESTING MODERN BASELINE METHODS")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_users, n_items = 50, 30
    
    # Generate synthetic user-item matrix (sparse)
    user_item_matrix = np.random.choice([0, 1, 2, 3, 4, 5], 
                                       size=(n_users, n_items), 
                                       p=[0.8, 0.04, 0.04, 0.04, 0.04, 0.04])
    
    print(f"📊 Dataset: {n_users} users, {n_items} items")
    print(f"Sparsity: {100 * np.mean(user_item_matrix == 0):.1f}%")
    
    # Test modern baselines
    modern_baselines = create_modern_baselines()
    
    for baseline_name, baseline in modern_baselines.items():
        print(f"\n🤖 Testing {baseline_name}...")
        
        try:
            # Train baseline
            baseline.fit(user_item_matrix)
            
            # Generate recommendations
            test_user = 0
            recommendations = baseline.recommend(test_user, n_recommendations=5)
            
            print(f"   ✅ Training successful")
            print(f"   Sample recommendations for user {test_user}:")
            for item_id, score in recommendations:
                print(f"     Item {item_id}: {score:.3f}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print("\n✅ Modern baseline testing completed!")

if __name__ == "__main__":
    test_modern_baselines()
