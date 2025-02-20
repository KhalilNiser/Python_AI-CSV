
# IMPORT_REQUIRED_DIRECTORIES
import random

# HUNT_THE_WUMPUS_CLASS
class HuntTheWumpus:
    
    # __init__() function
    def __init__( self ):
        
        self.caves = self.create_cavesMap()
        self.player_position = random.choice( list( self.caves.keys() ) )
        self.wumpus_position = random.choice( list( self.caves.keys() ) )
        
        while self.wumpus_position == self.player_position:
            self.wumpus_position = random.choice( list( self.caves.keys() ) )
            
        self.pits = random.sample( list( self.caves.keys() ), 2 )
        self.bats = random.sample( list( self.caves.keys() ), 2 )
        self.arrows = 3
        
        
    # create_cavesMap() function
    def create_cavesMap( self ):
        return {
            1: [ 2, 5, 8 ], 2: [ 1, 3, 10 ], 3: [ 2, 4, 12 ], 4: [ 3, 5, 14 ], 5: [ 1, 4, 6 ],
            6: [ 5, 7, 15 ], 7: [ 6, 8, 17 ], 8: [ 1, 7, 9 ], 9: [ 8, 10, 18 ], 10: [ 2, 9, 11 ],
            11: [ 10, 12, 19 ], 12: [ 3, 11, 13 ], 13: [ 12, 14, 20 ], 14: [ 4, 13, 15 ], 15: [ 6, 14, 16 ],
            16: [ 15, 17, 20 ], 17: [ 7, 16, 18 ], 18: [ 9, 17, 19 ], 19: [ 11, 18, 20 ], 20: [ 13, 16, 19 ]
        }
        
        
    # move_player() function
    def move_player( self, new_position ):
        if new_position in self.caves[ self.player_position ]:
            self.player_position = new_position
            self.check_hazards()
        else:
            print( "You Cannot Move There!" )
       
       
    # check_hazards() function
    def check_hazards( self ):
        
        if self.player_position == self.wumpus_position:
            print( "The Wumpus Got You! GAME_OVER!" )
            exit()
        elif self.player_position in self.pits:
            print( "You Fell Into a Bottomless Pit! GAME_OVER!" )  
            exit()
        elif self.player_position in self.bats:
            print( "Super Bats Will Carry You Over to Another Cave!" )
            self.player_position = random.choice( list( self.caves.keys() ) )
            self.check_hazards()
            
    
    # shoot_arrow() function
    def shoot_arrow( self, target_position ):
        
        if self.arrows > 0:
            self.arrows -= 1
            
            if target_position == self.wumpus_position:
                print( "You Killed the Wumpus! YOU_WON!" )
                exit()
            else:
                print( "YOU MISSED!" )
                
                if random.random < 0.5:
                    self.wumpus_position = random.choice( self.caves[ self.wumpus_position ] )
                    print( "The Wumpus has Moved!" )
                    
        else:
            print( "You Have No Arrows Left!" )
            
            
    # play() function
    def play( self ):
        
        print( "Welcome to Hunt the Wumpus!" )
        print( "Come on! Kill That Wumpus!" )
        
        while True:
            print( f"You are Inside cave: { self.player_position }!" )
            print( f"Adjacent Caves { self.caves[ self.player_position ] }" )
            
            action = input( "Move ( M cave# ) or Shoot ( S cave# )? " ).strip().upper()
            parts = action.split()
            
            if len( parts ) == 2 and parts[ 0 ] == 'M':
                self.move_player( int( parts[ 1 ] ) )
            elif len( parts ) == 2 and parts[ 0 ] == 'S':
                self.shoot_arrow( int( parts[ 1 ] ) )
            else:
                print( "Invalid Command!" )
                
                
if __name__ == "__main__":
    game = HuntTheWumpus()
    game.play()       
        
        