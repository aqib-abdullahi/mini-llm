from self_attention.linear_layer import Linear
from rectified_linear_unit import ReLU

class FeedForward:
    
    def __init__(self, emb_dim):
        
        self.embedding_dim = emb_dim
        self.hidden_dim = emb_dim * 4
        
        # linear
        self.linear1 = Linear(self.embedding_dim, self.hidden_dim)
        self.linear2 = Linear(self.hidden_dim, self.embedding_dim)
        self.relu = ReLU()
    
    def forward(self, embedding):
        
        output1 = self.linear1.forward(embedding)
        activated = self.relu.forward(output1)
        output2 = self.linear2.forward(activated)
        
        return output2