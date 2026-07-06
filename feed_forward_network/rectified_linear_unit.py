

class ReLU:
    
    def forward(self, values):
        
        output = []
        
        for value in values:
            if value <= 0:
                output.append(0)
            else:
                output.append(value)
        
        return output