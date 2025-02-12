
#               ---- IMPORT_REQUIRED_LIBRARIES ----
import torch
import torch.nn as nn
import torch.optim as optim

# Define a Neural Network Model
class NeuralNetwork( nn.Module ):
    def __init__( self ):
        super( NeuralNetwork, self ).__init__()
        # defines a Linear Layer that transforms the 
        # input data from (feature_number_value, 
        # feature_number_value)
        self.fc1 = nn.Linear( 10, 8 )
        self.fc2 = nn.Linear( 8, 2 )
        self.relu = nn.ReLU()
        
    def forward( self, x ):
        x = self.relu( self.fc1( x ) )
        x = self.fc2( x )
        return x
    
# Create an instance fo Neural Network
model = NeuralNetwork()
    # Define the "loss function"
    criterion = nn.CrossEntropyLoss()
    
    # Define the Optimizer
    optimizer = optim.SGD( model.parameters(), lr = 0.01 )
    
    # Sample Input Data
    # Reshaped to (1, 10)
    input_data = torch.randn( 1, 10 )
    
    # Perform Forward Pass
    output = model( input_data )
    
    # Perform Backward Pass and update the weights.
    optimizer.zero_grad()
    # Assuming the target class is 1
    loss = criterion( output, torch.tensor( [ 1 ] ) )
    loss.backward()
    optimizer.step()
    
    # Print the updated model parameters
    print( "Updated Model Parameters:" )
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print( name, param.data )
            
        # Printing the loss value
        print( "Loss:", loss.item() )
        
    def calculate_accuracy( output, target ):
        _, predicted = torch.max( output, 1 )
        accuraccy = ( predicted == target ).sum().item() / target.size( 0 )
        return accuraccy
    
    # Calculate and print the accuracy: 
    # Assuming the target class is 1
    target = torch.tensor( [ 1 ] )
    accuracy = calculate_accuracy( output, target )
    print( "Accuracy:", accuracy )
             
    
    
    

