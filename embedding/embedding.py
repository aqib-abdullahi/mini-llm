import random
import math


class Embedding:
    
    def __init__(self, vocab_size, embedding_dim):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.weight = []
        
        for _ in range(vocab_size):
            embedding = []
            
            for _ in range(embedding_dim):
                embedding.append(random.uniform(-1.0, 1.0))
            self.weight.append(embedding)
    
    def forward(self, token_ids):
        
        embedding_vectors = []
        for token_id in token_ids:
            embedding_vectors.append(self.weight[token_id])
        
        return embedding_vectors

    def rotate(self, x, y, theta):
        
        x_new = (x * math.cos(theta)) - (y * math.sin(theta))
        y_new = (x * math.sin(theta)) + (y * math.cos(theta))
        
        return x_new, y_new
    
    def apply_rope(self, embedding, theta):
        
        rotated_embedding = []
        i = 0
        
        while i < len(embedding):
            x = embedding[i]
            y = embedding[i + 1]
            
            x_new, y_new = self.rotate(x, y, theta)
            
            rotated_embedding.append(x_new)
            rotated_embedding.append(y_new)
            
            i+=2
            
        return rotated_embedding