
# IMPORT_UUID: Python Built-in_module UUID: "Universally Unique ID": 
# Popular in SQL database. It gives each item it's own unique ID-Number.
import uuid
# To load my inventory
import csv
 
#               ---- INVENTORY_CLASS ----
# InventoryItem Class: 
class InventoryItem:
    # This function stores data about the item
    def __init__( self, item_id, name, quantity, price ):
        # Item property
        self.item_id = item_id
        self.name = name
        self.quantity = quantity
        self.price = price
        self.sku = str( uuid.uuid4() )
        
    # Display item information. Using a String format fashion. 
    # NOTE: The ".2f", rounds down the price
    def display_info( self ):
        print( f"Item ID: { self.item_id }, Name: { self.name }, Quantity: { self.quantity }, Price: ${ self.price:.2f }" )
    
    
    
    
class InventoryManager:
    def __init__( self, file ):
        # Create a CSV file and an empty array "inventory" list
        self.file = file
        self.inventory = [] 
        # This will store my automatically generated ID. 
        # I have currently set it to "0"
        self.next_item_id = 0
    
    # Load all pre-existing/existing inventory
    def load_inventory( self ):
        # Try/Except(Catch)-block Exception. As precautionary:
        # Might not needed, but just incase. Since Im dealing
        # with lots of overwhelming and unfamiliar information.
        
        # Try-block
        try:
            # Open and read (r) from the file I called "file"
            with open( self.file, "r" ) as file:
                # Create an object of the class "csv.DictReader()":
                # Named "reader". To store/read from the file
                reader = csv.DictReader( file )
                # Create a list of informationfor every item I 
                # want stored inside my list. Integers, floats 
                # and Stirings
                self.inventory = [ InventoryItem(
                    int(row[ "ItemID" ] ),
                    row[ "Name" ],
                    int( row[ "Quantity" ] ),
                    float( row[ "Price" ] ), )
                                # For every row in my reader
                                  for row in reader ]
                # After I add an item to the list. I want to take 
                # that next item ID. Im saying that's now equal to 
                # the lngth of my current list, which is inventory. 
                # but now I want to add one to it, by incrementing. 
                # And that will allow python to keep counting next...
                self.next_item_id = len( self.inventory ) + 1
        # Catch-block("except" in python)
        except FileNotFoundError:
            # NOTE: It will create a file for you, if needed 
            print( "No Such File Exists...!" )
    
    def save_inventory( self ):
        # Open and write (w), to the file I just 
        # created and named file. 
        with open( self.file, "w", newline="" ) as file:
            fieldnames = [ "ItemID", "Name", "Quantity", "Price", "SKU" ]
            writer = csv.DictWriter( file, fieldnames=fieldnames )
            writer.writeheader()
            writer.writerows( {
                "ItemID": item.item_id,
                "Name": item.name,
                "Quantity": item.quantity,
                "Price": item.price,
                "SKU": item.sku,
            }
                            #  For every item in my entire inventory list
                             for item in self.inventory )
            
    
    def add_item( self, name, quantity, price ):
        item = InventoryItem( self.next_item_id, name, quantity, price )
        # Now, take my list (inventory), and append to it the 
        # tem/object just created. Add one, by incrementing 
        # and move on to the next
        self.inventory.append( item )
        self.next_item_id += 1
        # Save that item to my CSV File
        self.save_inventory()
    
    def display_inventory( self ):
        # If my inventory list is empty, display (print())
        if not self.inventory:
            print( "List is Empty...!" )
        else:
            # For every item inside my invemtory 
            for item in self.inventory:
                # Calling on my "display_info()" method
                item.display_info()
        
    
    def delete_item_by_id( self, item_id ):
        # Take my list, store into my inventory. I want to have an item in 
        # this list. For every item in my inventory. If, the current item 
        # ID, is "Not" equal to the item ID I'm looking for. I want to add 
        # it to the list. This will create a "new" list of every single 
        # item except the item that has this ID (item_id) parameter
        # and argument inside my delete_item_by_id() method
        self.inventory = [ item for item in self.inventory if item.item_id != item_id ]
        # There is going to be a new inventory. Save that 
        # inventory. Now I have a new inventory list
        self.save_inventory()
    # Filter by maximum price I'm looking for
    def filter_items( self, max_price ):
        return[ item for item in self.inventory if item.price <= max_price ]
    
def main():
    store = InventoryManager( "inventory.csv" )
    store.load_inventory()
    
    while True:
        print( "\n##### E-Commerce System #####" )
        print( "1.  ADD an Item" )
        print( "2.  Display Inventory" )
        print( "3.  Filter Items by Price" )
        print( "4.  Delete an Item by ID" )
        print( "5.  Save Inventory" )
        print( "6.  Exit" )
        
        option = input( "Enter Your Choice  (1-6): " )
        # ADD_AN_ITEM
        if option == "1":
            name = input( "Enter Item Name: " )
            quantity = int( input( "Enter the Amount of Items: " ) )
            price = float( input( "Enter Item Price: " ) )
            store.add_item(name, quantity, price)
            
        # DISPLAY_INVENTORY
        elif option == "2":
            store.display_inventory()
            
        # FILTER_ITEMS_BY_PRICE
        elif option == "3":
            search = float(input( "Enter a Max Price: " ) )
            filter_items = store.filter_items( search )
            print( f"Items Less Than {search}: " )
            # For every item inside my "filter_items" list
            for item in filter_items:
                item.display_info()
                
        # DELETE_AN_ITEM_BY_ID
        elif option == "4":
            item_id_num_to_delete = int(input( "Enter the ID of the Item to Delete: " ) )
            store.delete_item_by_id( item_id_num_to_delete )
            print( "Successfully Delted Item!")
        
        # SAVE_INVENTORY
        elif option == "5":
            store.save_inventory()
            print( "Saved to CSV File!" )
        
        # BREAK/EXIT_FROM_INVENTORY
        elif option == "6":
            print( "Goodbye...!" )
            break
        
        else:
            print( "Invalid Entry! Please Enter an Option Beetween 1-6!" )
            
if __name__ == "__main__": 
    main()
        
    