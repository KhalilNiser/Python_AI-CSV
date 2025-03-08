
#       ---- IMPORT_REQUIRED_LIBRARY ----
import matplotlib.pyplot as plt


# Sample Data
x_values = range( 10 )
y_values = [ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 ]

# Create the plot
plt.plot( x_values, y_values )


# Add labels and title
plt.xlabel( 'X-axis' )
plt.ylabel( 'Y-axis' )
plt.title( "Simple Line Graph" )

# Display the plot
plt.show()
