
#               ---- BINARY_SEARCH_TREE ----
# Binary Tree is nothing but a regular tree with a constraint that
# it has at most "two" child Nodes. The maximum number of child 
# nodes that a particular node can have is two. If I was to say an 
# example of an Electronics Store, it sells laptops, cellphones, 
# TV's. Cellphones: It has IPhones, Google Pixel, Vivo, Samsung, ... 
# etc. That is Not considered a Binary Tree, that's more of a General 
# Tree. 
# Now, Binary Search Tree (BST), is a special case of Binary Tree where 
# the elements have some kind of order. The order is: All the Nodes on 
# the left side of the particular Node has a value less than a particular 
# Node. Let's take the number 15, for instance. All the elements to the 
# left of the tree are less than 15, while all the elements to right of 
# the tree, are greater than or equal to 15. And that method applies to 
# every Node on the tree. Under this property elements "cannot", be 
# duplicated. 
# If you were to search for the node 14, you will start by looking at 15. 
# 14, is less than 15, hence you are sure that 14, would be on the left 
# sub tree, ..., etc. Until you find 14 (If it exists on the list). That 
# way you are reducing your search base by half. The same would apply to 
# insert. If you were to insert the data element 13, well, you know 13 is 
# less than 15. So 13, goes on the left of the Node Tree. 27, is greater 
# than 15, so 27 goes on the right side of the Node Tree.
# 
# If you have 8 elements, going one by one, that would "linear O(n)". It 
# would take all day for larger data. 
# 
# Search Example: Order of Operation and Time Complexity:
# Every iteration I'm reducing the search/time by 1/2. 
# n = 8:       8 → 4 → 2 → 1
# 3 iterations 
# log2 8 = 3 
# Time Complexity O(log n) 

# To search through the binary tree, there are two search methods 
# Breadth-First-Search, and Depth-First-Search (in-order traversal, 
# pre-order traversal, and post-order traversal), aka. Traversal 
# techniques.  meaning: How do you traverse through your binary tree 
# to find the data elemenet. 
# (NOTE: When we say in-order, pre-order, or post-order, we are refering 
# to the "base node"). In-order, you first visit the left sub-tree, then 
# the root-node, then the right sub-tree. Pre-order you visit the root-Node 
# first, then left sub-tree, then right sub-tree. Post-order, you visit the 
# left sub-tree, right sub-tree, then the root-node. In all three techniques 
# I'm using the "recursive algorithm". HINT: One thing to remember: Where do 
# you place your root-node first. In-order (in-between), pre-order (beginning), 
# post-order (last).
# 
# Traversal Techniques: 
# In-order traversal: [ 7, 12, 14, 15, 20, 23, 27, 88 ] 
# Pre-order traversal: [ 15, 12, 7, 14, 27, 20, 23, 88 ] 
# Post-order traversal: [ 7, 14, 12, 23, 20, 88, 27, 15 ] 

# In this Python code program (Python) I will be focusing on the in-order 
# traversal technique. 


#               ---- IN-ORDER-TRAVERSAL_TECHNIQUE ----
# BinarySearchTreeNode class
class BinarySearchTreeNode:
    # Base Tree
    def initial_tree( self, data ):
        self.data = data
        self.left = None
        self.right = None
        
    # This node could be root node. Or just about any node in the tree.
    # Whenever I want to add a child with a value data. First I need to 
    # check. If data i.e. to self data and already exists
    def insert_child_data( self, data ):
        if data == self.data:
            # Don't do anything, since no duplicate is allowed
            return
        
        if data < self.data:
            # Add data to left sub-tree
            if self.left:
                self.left.insert_child_data( data )
            
            else:
                self.left = BinarySearchTreeNode( data )
        else:
            # Add data to right sub-tree
            if self.right:
                self.right.insert_child_data( data )
            else:
                self.right = BinarySearchTreeNode( data )
                
    def in_order_traversal( self ):
        elements = []
        # Visit left tree
        if self.left:
            elements += self.left.in_order_traversal()
            
        # Visit base node
        elements.append( self.data )
        
        # Visit right tree
        if self.right:
            elements += self.right.in_order_traversal()
            
        return elements
    

# Function build_tree takes elements as an input
def build_tree( elements ):
    root_node = BinarySearchTreeNode( elements[ 0 ] )
    
    for i in range( 1, len( elements ) ):
        # Add child method
        root_node.insert_child_data( elements[ i ] )
        
    return root_node
    
if __name__ == '__main__':
    numbers = [ 17, 4, 1, 20, 9, 23, 18, 34 ]
    
    numbers_tree = build_tree( numbers )
    
    print( numbers_tree.in_order_traversal() )
            
        
            
        
        
