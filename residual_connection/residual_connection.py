
class ResidualConnection:
    
    def forward(self, original, layer_output):
        
        output = []
        
        if len(original) != len(layer_output):
            raise ValueError(
                "Residual inputs must have the same dimentsion"
            )
        
        for embedding, layer_out in zip(original, layer_output):
            output.append(embedding + layer_out)
        
        return output