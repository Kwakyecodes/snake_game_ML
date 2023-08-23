import pygame
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from characters_small import (
    # Characters
    Snake, 
    Food, 
    Explosive,
    
    # Display variables
    win_height,
    win_width,
)

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

def make_next_move(model, matrix):
    # Predicted move converter
    converter = {
        0: "UP",
        1: "RIGHT",
        2: "DOWN",
        3: "LEFT"
    }
    
    # Add extra dimension to matrix
    matrix = np.expand_dims(matrix, axis=(0, -1)) # resulting shape will be (1, 60, 80, 1)
    
    # Now you can use your model to make predictions
    predictions = model.predict(matrix)
    
    # The output is a batch of logits for each class, so let's take the first (and only) output
    output = predictions[0]
    
    # Convert logits to probabilities
    probabilities = tf.nn.softmax(output).numpy()

    # Get class with the highest probability as models class
    predicted_class = np.argmax(probabilities)
    return converter[predicted_class]
   
def game_loop():
    """Main game loop"""

    # Constants
    SNAKE_SPEED = 20
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
    snake = Snake()
    food = Food()
    explosive = Explosive()
    
    # Recreate the exact same model architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6, 8, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))  
    model.add(layers.Flatten())  # Flatten the feature map
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # Load the weights
    model.load_weights('cnn_weights.h5')

    while True:
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
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        # Predict the next move with weights of cnn model
        new_snake_direction = make_next_move(model=model, matrix=matrix)
        
        # Update snake direction with new move
        snake.direction = new_snake_direction

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
        # We want to see how data performs at one level
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
        
        # REMOVING EXPLOSIVES TO MAKE GAME SIMPLER
        # # Logic for spawning explosives
        # EXPLOSIVE_TIMER += 1
        # if EXPLOSIVE_TIMER % 20 == 0: 
        #     explosive.spawn_explosive(matrix=matrix)
            
        # # Logic for timing out explosives
        # if EXPLOSIVE_TIMER % 200 == 0:
        #     explosive.destroy_explosive(0)
            
        # Drawing snake, good food and explosives
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
    print(f"SCORE: {score}")
    pygame.display.update()
    pygame.time.delay(5000) # Display final score for 5 seconds
    
    # End game
    pygame.quit()


# Command to run game
if __name__ == "__main__":
    game_loop()