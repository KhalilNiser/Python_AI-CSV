
import pandas as pd

data = pd.read_csv( r'grades.csv' )


#               ---- AVG_GRADE ----
# Calculates the average grade for each student
grade_avg = data.groupby( 'Student Name' )[ 'Grade' ].mean()

# Calculates and displays the "minimum" grade in the class
min_grade = data[ 'Grade' ].min()
print( "Lowest Grade in the Class: ", min_grade )

# Calculates and displays the "maximum" grade in the class
max_grade = data[ 'Grade' ].max()
print( "Lowest Grade in the Class: ", max_grade )

# alculates and displays the class average overall
class_avg = data[ 'Grade' ].mean()
print( "Average Grade of the Class: ", class_avg )


#               ---- PASS_FAIL ----
# New column (Pass/Fail): Indicates whether each student Passed or failed 
data[ 'Pass/Fail' ] = data[ 'Grade' ].apply( lambda X: 'Pass' if X >= 60 else 'Fail' )
print( data )

# Counts the number of students who passed and failed
pass_count = data[ data[ 'Pass/Fail'] == 'Pass' ].shape[ 0 ]
fail_count = data[ data[ 'Pass/Fail'] == 'Fail' ].shape[ 0 ]

print( "Number of Students Passed: ", pass_count )
print( "Number of Students Failed: ", fail_count )
