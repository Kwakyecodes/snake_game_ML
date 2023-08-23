"""
USE DIJKSTA'S ALGORITHM (WITH HUMAN AID) TO COLLECT DATA ON THE
BEST DIRECTIONS FOR THE SNAKE TO LOCATE THE FOOD

NOTE:
1. ONE DATA POINT IS A MATRIX OF SIZE 60 BY 80
2. LABEL IS A ONE-DIMENSIONAL MATRIX 
3. THERE ARE 4 CLASSES: 0 -> UP, 1 -> RIGHT, 2 -> DOWN, 3 -> LEFT.
"""

import pygame
import pickle
import math
import numpy as np
from characters_small import (
    # Characters
    Snake, 
    Food, 
    Explosive,
    
    # Display variables
    win_height,
    win_width,
)

def labeler(direction):
    # Change direction(string) to discrete value for labeling
    converter = {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
    }
    return converter[direction]

def update_matrix(curr_matrix, snakes, food, explosives):
    """Update the position of characters in matrix"""
    
    # Update snake's head and body
    for ind, pos in enumerate(snakes):
        x, y = pos[0]//10, pos[1]//10
        if ind == 0: # snake's head is now 3
            curr_matrix[y][x] = 11
        else: # snake's body is set to 2
            curr_matrix[y][x] = 2
            
    # Update explosives in matrix
    for pos in explosives: 
        # Explosives are also set to 2
        x, y = pos[0]//10, pos[1]//10
        curr_matrix[y][x] = 2 
    
    # Update food in matrix
    x, y = food[0]//10, food[1]//10
    curr_matrix[y][x] = 6 
    
    return curr_matrix

def calc_cost(A, B):
    """Takes two points A and B and calculates cost of getting from A to B"""
    cost = abs(A[1] - B[1]) + abs(A[0] - B[0])
    return cost

def make_next_move(matrix, head, length_of_snake, food, curr_direction):
    """Select next move of snake"""
    
    h, w = len(matrix), len(matrix[0])
    directions = [ 
        {"name": "LEFT", "value": (0, -1), "opposite": "RIGHT"},
        {"name": "UP", "value": (-1, 0), "opposite": "DOWN"},
        {"name": "RIGHT", "value": (0, 1), "opposite": "LEFT"},
        {"name": "DOWN", "value": (1, 0), "opposite": "UP"}, 
        # Opposite key value pair of directions ensures that
        # Assistant does not move into its body
    ]
    head = (head[1]//10, head[0]//10) # Scale position of head and reverse for numpy
    food = (food[1]//10, food[0]//10) # Scale position of food and reverse for numpy
    n = 1 # number of items in heap
    heap = [
        {"path": [head], "direction": curr_direction}
    ]
    move = None
    while heap:
        # Sort heap in descending order based on proximity to food
        heap.sort(key=lambda x: calc_cost(x["path"][-1], food), reverse=True)
    
        path_obj = heap.pop()
        n -= 1 # Reduce size of heap since the last item has been popped
        curr = path_obj["path"][-1]
        curr_direction = path_obj["direction"]
    
        # Stop searching and make move when food is found
        if matrix[curr[0]][curr[1]] == 6:
            move = path_obj["path"][1] # Set move to next step from head   
            break
        
        if n > 0 and math.log10(n) >= 3:
            move = input("Press w, a, s, d for direction: ")
            break
        
        # Find next possible moves
        for d in directions:
            
            # Prevent snake from going the opposite direction
            if d["opposite"] == curr_direction:
                continue
            
            y, x = d["value"]
            next_y, next_x = curr[0] + y, curr[1] + x
            nextt = (next_y, next_x)
            
            # Check if path has already been visited. This is important because each time this
            # function is called, every path from head to the food is determine; visiting an already
            # visited path means snake crashes into itself. Ideally, the matrix could be changed to
            # reflect this change in the snake's (temporary) position but that's too much work and I'm lazy boi :o
            if nextt in path_obj["path"]:
                continue
            
            # Skip moves where snake crashes into wall
            if (next_y not in range(0, h)) or (next_x not in range(0, w)):
                continue
            
            # Skip moves where snake crashes into explosives or itself
            if matrix[next_y][next_x] == 2:
                continue
            
            new_heap_obj = {
                "path": path_obj["path"] + [nextt],
                "direction": d["name"]
                } 
            heap.append(new_heap_obj)
            n += 1 # Increase size of head each time an item is added
           
    # Handle human aided move
    if type(move) == str:
        if move == "w":
            return "UP"
        elif move == "d":
            return "RIGHT"
        elif move == "s":
            return "DOWN"
        else: # move == "a"
            return "LEFT"
        
    if move:        
        # Select direction 
        move_pos = (move[0] - head[0], move[1] - head[1])
        for d in directions:
            if d["value"] == move_pos:
                move = d["name"] # Change move to direction name
                break   
    return move
   
def game_loop():
    """Main game loop"""

    # Constants
    SNAKE_SPEED = 24
    EXPLOSIVE_TIMER = 0
    SNAKE_COLOR = (255, 255, 255)
    FOOD_COLOR = (0, 0, 255)
    EXPLOSIVE_COLOR = (255, 0, 0)
    BG_COLOR = (0, 0, 0)
    TEXT_COLOR = (255, 255, 255)

    # Set up game clock
    clock = pygame.time.Clock()

    # Initialize Pygame
    pygame.init()
    win = pygame.display.set_mode((win_width, win_height))

    score = 1
    snake = Snake() # Spawns snake at level 1
    food = Food() # Initiates with no food at random position; could be position of snake but what are the chances :) 
    explosive = Explosive() # Initiates with zero explosives

    
    m = 0  # Space of the training set
    
    while True:
        # End game when enough data has been collected
        if m == 10000:
            break
            
        # Create new matrix
        """
        Values in matrix
        0 - empty cell
        1 - food
        2 - dangerous cells (snake's body and explosives)
        """
        matrix = np.zeros((win_height//10, win_width//10))
    
        # Update cells of matrix
        matrix = update_matrix(matrix, snake.body, food.position, explosive.explosives)
        
        # Collect new data point into pickle file
        with open('training_data.pkl', 'ab') as f:  # 'ab' mode for appending in binary mode
            pickle.dump(matrix, f)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        new_snake_direction = make_next_move(
            matrix=matrix, 
            head=snake.body[0], 
            length_of_snake=len(snake.body),
            food=food.position, 
            curr_direction=snake.direction
        )
        if new_snake_direction:
            snake.direction = new_snake_direction
        else: # End game because no valid move was found
            break

        # Collect new label into pickle file
        with open('labels.pkl', 'ab') as f:  # 'ab' mode for appending in binary mode
            pickle.dump(labeler(direction=new_snake_direction), f)
        
        m += 1  # increase m by 1 after every new data collected

        # Move the snake
        snake.grow() 
        
        # Check if snake bit itself or has hit wall and end game if it has
        if (snake.body[0] in snake.body[1:] or
            snake.body[0][0] not in range(0, win_width) or
            snake.body[0][1] not in range(0, win_height)):
            break
            
        # Logic for eating food and spawning new one
        if snake.body[0] == food.position:
            score += 1
            food.spawn_new_food(matrix=matrix)
            
        # Remove last segment whether food is eaten or not. 
        # We need multiple data on just level 1
        snake.shrink()
            
        # Update cells of matrix (effect change in food's position)
        matrix = update_matrix(matrix, snake.body, food.position, explosive.explosives)
            
        # Logic for running into bombs  
        if snake.body:
            if snake.body[0] in explosive.explosives:
                for i, explo in enumerate(explosive.explosives):
                    if explo == snake.body[0]:
                        score -= 1
                        explosive.destroy_explosive(i)
                        snake.shrink()
            
        # Check if snake is dead and end game if so
        if len(snake.body) == 0:
            break
        
        win.fill(BG_COLOR)
        for segment in snake.body:
            pygame.draw.rect(win, SNAKE_COLOR, pygame.Rect(segment[0], segment[1], snake.size, snake.size))
        pygame.draw.rect(win, FOOD_COLOR, pygame.Rect(food.position[0], food.position[1], food.size, food.size))
        if explosive.explosives:
            for explo in explosive.explosives:
                pygame.draw.rect(win, EXPLOSIVE_COLOR, pygame.Rect(explo[0], explo[1], explosive.size, explosive.size))
                
        pygame.display.update()
        clock.tick(SNAKE_SPEED)
                
    # Display final scores
    print(score)
        
    # End game
    pygame.quit()


# Command to run game
if __name__ == "__main__":
    game_loop()