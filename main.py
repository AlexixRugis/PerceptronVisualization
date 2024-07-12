import pygame
import random
import numpy as np

from neuro_layers import LinearLayer, SigmoidActivator, TanhActivator, SequentialModel

WIDTH = 720
HEIGHT = 720

CIRCLE_SIZE = 20

FIRST_CLASS_COLOR = (247, 107, 15)
FIRST_CLASS_NETWORK_COLOR = (255, 133, 133)
SECOND_CLASS_COLOR = (15, 29, 247)
SECOND_CLASS_NETWORK_COLOR = (100, 107, 211)

DATASET_SAVE_FILE = 'data.txt'
DATASET_SIZE = 30
DATASET = []

LEARNING_RATE = 0.03
BATCH_SIZE = DATASET_SIZE

NETWORK = SequentialModel([
    LinearLayer(2, 5),
    TanhActivator(),
    LinearLayer(5, 5),
    TanhActivator(),
    LinearLayer(5, 2),
    TanhActivator()
])
NETWORK.randomize_weights()

def generate_dataset():
    for i in range(DATASET_SIZE//2):
        DATASET.append((random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 0))
    for i in range(DATASET_SIZE//2):
        DATASET.append((random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 1))
        
def draw_background(window, render_size = 10):
    rect = window.get_rect()
    
    for x in range(0, rect.width, render_size):
        u = (x + render_size / 2) / (rect.width - 1)
        for y in range(0, rect.height, render_size):
            v = (y + render_size / 2) / (rect.height - 1)
            value = NETWORK(np.array([[u, v]]).T)
            if value[0] > value[1]:
                color = FIRST_CLASS_NETWORK_COLOR
            else:
                color = SECOND_CLASS_NETWORK_COLOR
            pygame.draw.rect(window, color, (rect.left + x, rect.top + y, render_size, render_size))
            
def draw_dataset(window):
    for row in DATASET:
        pygame.draw.circle(window, 
                           (0, 0, 0), 
                           (row[0] * WIDTH, row[1] * HEIGHT), CIRCLE_SIZE)
        pygame.draw.circle(window, 
                           FIRST_CLASS_COLOR if row[2] == 0 else SECOND_CLASS_COLOR, 
                           (row[0] * WIDTH, row[1] * HEIGHT), CIRCLE_SIZE - 3)

def get_circle_under_mouse(mouse_pos):
    for index, row in enumerate(DATASET):
        screen_pos = (row[0] * WIDTH, row[1] * HEIGHT)
        
        if (screen_pos[0]-mouse_pos[0])*(screen_pos[0]-mouse_pos[0]) + (screen_pos[1]-mouse_pos[1])*(screen_pos[1]-mouse_pos[1]) < CIRCLE_SIZE * CIRCLE_SIZE:
            return index
        
    return -1

def save_dataset():
    with open(DATASET_SAVE_FILE, 'w') as f:
        f.write(str(len(DATASET)) + '\n')
        for row in DATASET:
            f.write(f'{row[0]} {row[1]} {row[2]}\n')
            
def load_dataset():
    try:
        with open(DATASET_SAVE_FILE, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                x, y, c = f.readline().split()
                DATASET.append((float(x), float(y), int(c)))
            
        return True
    except:
        return False
            
def init_dataset():
    if not load_dataset():
        generate_dataset()
        save_dataset()

def main():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF|pygame.HWSURFACE)
    pygame.display.set_caption("AI visuals")

    canvas = pygame.Surface((WIDTH, HEIGHT))

    init_dataset()

    draging_index = -1
    learn_index = 0
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if draging_index == -1:
                    draging_index = get_circle_under_mouse(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                if draging_index >= 0:
                    draging_index = -1
                    save_dataset()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                NETWORK.randomize()
        
        # update
        
        if draging_index >= 0:
            mouse_pos = pygame.mouse.get_pos()
            DATASET[draging_index] = (mouse_pos[0] / WIDTH, mouse_pos[1] / HEIGHT, DATASET[draging_index][2])
            
        # learn
        
        for i in range(BATCH_SIZE):
            d_x, d_y, d_r = DATASET[learn_index]
            if d_r == 0:
                NETWORK.fit(np.array([[d_x, d_y]]).T, np.array([[1.0, 0.0]]).T, LEARNING_RATE)
            else:
                NETWORK.fit(np.array([[d_x, d_y]]).T, np.array([[0.0, 1.0]]).T, LEARNING_RATE)
            learn_index = (learn_index + 1) % len(DATASET)
            
        # draw
        canvas.fill(0)
        
        draw_background(canvas, 10)
        draw_dataset(canvas)
        
        window.blit(canvas, (0, 0))
        pygame.display.flip()

    pygame.quit()
    exit()
    
if __name__ == '__main__':
    main()