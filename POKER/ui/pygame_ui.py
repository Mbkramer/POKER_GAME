from engine.game_state import GamePhase
from core.table_state import TableState
from core.player import Player
from core.deck import Deck
from core.card import HAND_RANK_NAMES, HAND_RANKS
from core.player_action import ActionType, PlayerAction
from engine.hand_controller import HandController
from core.hand_evaluator import HandEvaluator

import pygame as pg
import sys
import os
import math
import time
from typing import Optional

# ---------- pygame scene variables ----------

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
WINDOW_CENTER_X = 500
WINDOW_CENTER_Y = 300
FPS = 60

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)
GOLD = (255, 215, 0)
TRANSPARENT_GOLD = (255, 215, 0, 128)
TRANSPARENT_BLUE = (0, 0, 255, 128)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)
DARK_RED = (139, 0, 0)
BLACK = (0, 0, 0)

#Table
BACKGROUND_IMAGE_PATH = "GAME_IMAGES/TABLE/Poker_Table_Dark.png"
MAX_PLAYERS = 6

#Text
POT_FONT = "monaco"
MONEY_FONT = "monaco"
PLAYER_FONT = "arial"
INPUT_FONT = "monaco"

#Cards
CARDS_FOLDER_PATH = "GAME_IMAGES/CARDS/"
CARD_WIDTH = 71 
CARD_HEIGHT = 94

#Decks
COMMUNITY_CARDS_PLACEMENT_Y = 261
COMMUNITY_CARDS_PLACEMENT_X = 321

#Pot placement
POT_PLACEMENT_X = 418
POT_PLACEMENT_Y = 412

#Dealer placement
DEALER_PLACEMENT_X = 650
DEALER_PLACEMENT_Y = 125

#Input Window
INPUT_WINDOW_HEIGHT = 300
INPUT_WINDOW_WIDTH = 350
INPUT_WINDOW_PLACEMENT_X = WINDOW_CENTER_X - INPUT_WINDOW_WIDTH/2
INPUT_WINDOW_PLACEMENT_Y = WINDOW_CENTER_Y - INPUT_WINDOW_HEIGHT/2

#Game Over Window
GAME_OVER_HEIGHT = 400
GAME_OVER_WIDTH = 800
GAME_OVER_PLACEMENT_X = 100
GAME_OVER_PLACEMENT_Y = 100

# Graph parameters
GRAPH_START_X = GAME_OVER_PLACEMENT_X + GAME_OVER_WIDTH/8
GRAPH_START_Y = GAME_OVER_PLACEMENT_Y + GAME_OVER_HEIGHT - 25
GRAPH_HEIGHT = 75
X_SPACING = 10

#Hud Window
HUD_WINDOW_PLACEMENT_X = 200
HUD_WINDOW_PLACEMENT_Y = -100
HUD_WINDOW_HEIGHT = 300
HUD_WINDOW_WIDTH = 350

COLOR_INACTIVE = pg.Color('lightskyblue3')
COLOR_ACTIVE = pg.Color('dodgerblue2')

#Buttons
NUM_PLAYERS_BUTTON_WIDTH = 25
NUM_PLAYERS_BUTTON_HEIGHT = 25
NUM_PLAYERS_BUTTON_X = INPUT_WINDOW_PLACEMENT_X + 2.5*NUM_PLAYERS_BUTTON_WIDTH
NUM_PLAYERS_BUTTON_Y = INPUT_WINDOW_PLACEMENT_Y + 150

#Start buttons
START_BUTTON_WIDTH = 100
START_BUTTON_HEIGHT = 50
START_BUTTON_PLACEMENT_X = INPUT_WINDOW_PLACEMENT_X + INPUT_WINDOW_WIDTH/2 - START_BUTTON_WIDTH/2
START_BUTTON_PLACEMENT_Y = INPUT_WINDOW_PLACEMENT_Y + INPUT_WINDOW_HEIGHT

#Hud Buttons
HUD_PLAYERS_BUTTON_X = HUD_WINDOW_PLACEMENT_X+20
HUD_PLAYERS_BUTTON_Y = 80
HUD_BUTTON_WIDTH = 50
HUD_BUTTON_HEIGHT = 25

#log player action
LOG_BUTTON_WIDTH = 100
LOG_BUTTON_HEIGHT = 50
LOG_BUTTON_PLACEMENT_X = HUD_WINDOW_PLACEMENT_X + HUD_WINDOW_WIDTH/2 - LOG_BUTTON_WIDTH/2
LOG_BUTTON_PLACEMENT_Y = HUD_WINDOW_PLACEMENT_Y + HUD_WINDOW_HEIGHT

NEW_ROUND_BUTTON_WIDTH = 100
NEW_ROUND_BUTTON_HEIGHT = 50
NEW_ROUND_BUTTON_PLACEMENT_X = HUD_WINDOW_PLACEMENT_X+HUD_WINDOW_WIDTH-NEW_ROUND_BUTTON_WIDTH/2
NEW_ROUND_BUTTON_PLACEMENT_Y = HUD_WINDOW_PLACEMENT_Y+HUD_WINDOW_HEIGHT-NEW_ROUND_BUTTON_HEIGHT/2

#Input Boxs
INPUT_BOX_WIDTH = 75
INPUT_BOX_HEIGHT = 10

#Players
PLAYER_WIDTH = 125
PLAYER_HEIGHT = 100
PLAYER_RADIUS = 32
PLAYER_COLOR = GRAY

ELLIPSE_PLACEMENT_WIDTH = 900 #900
ELLIPSE_PLACEMENT_HEIGHT = 500 #500
ELLIPSE_CENTER_X = WINDOW_CENTER_X - (PLAYER_WIDTH/2)
ELLIPSE_CENTER_Y = WINDOW_CENTER_Y - (PLAYER_HEIGHT/2)
ELLIPSE_PLACEMENT_X = ELLIPSE_CENTER_X - (ELLIPSE_PLACEMENT_WIDTH) / 2
ELLIPSE_PLACEMENT_Y = ELLIPSE_CENTER_Y - (ELLIPSE_PLACEMENT_HEIGHT) / 2

#Chips
CHIPS_FOLDER_PATH = "GAME_IMAGES/CHIPS/"

#Values
WHITE_CHIP = 1
RED_CHIP  = 5
GREEN_CHIP = 25
GRAY_CHIP = 100
BLUE_CHIP = 500
YELLOW_CHIP = 1000
LIGHT_GRAY_CHIP = 5000
BIEGE_CHIP = 10000
LIGHT_BLUE_CHIP = 25000
PINK_CHIP = 50000
PURPLE_CHIP = 100000

CHIP_VALUES = (100000, 50000, 25000, 10000, 5000, 1000, 500, 100, 25, 5, 1)

TOP_CHIP_WIDTH = 25 
TOP_CHIP_HEIGHT = 25

FLAT_CHIP_WIDTH = 25
FLAT_CHIP_HEIGHT = 12

class InputBox:
    def __init__(self, x, y, width, height, input_font, text=""):

        self.background_rect = pg.Rect(x-5, y-5, width+10, height+10)
        self.rect = pg.Rect(x, y, width, height)
        self.input_font = input_font
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = input_font.render(text, True, BLACK)
        self.active = False
        self.stored_input = 0

    def handle_event(self, event):

        if event.type == pg.MOUSEBUTTONDOWN:
            
            click_x, click_y = event.pos

            if ((self.rect.x <= click_x <= self.rect.x+self.rect.width) and (self.rect.y <= click_y <=self.rect.y+self.rect.height)):
                self.active = not self.active
            else:
                self.active = False

        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    # When 'Enter' is pressed, store the text and reset the input box
                    try:
                        store = int(self.text)
                        if store > 0:
                            self.active = False
                            self.color = COLOR_INACTIVE
                            self.stored_input = store
                    except:
                        self.text = "0"
                    
                    #self.text = '' # Clear the box after storing
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1] # Remove the last character
                else:
                    self.text += event.unicode # Add the key's unicode character
                    try: int(self.text)
                    except: self.text = self.text[:-1]

                self.txt_surface = self.input_font.render(f"${self.text}", True, BLACK)

        self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE

    def update(self):
        # Resize the box if the text is too long
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        
        # Draw the box rectangle
        pg.draw.rect(screen, self.color, self.background_rect)  
        pg.draw.rect(screen, self.color, self.rect) 

        # Blit the text
        screen.blit(self.txt_surface, (self.rect.x, self.rect.y))


class Button:
    def __init__(self, text, x, y, width, height, color, hover_color):
        self.text = text
        self.x = x
        self.y = y
        self.active = False
        self.width = width
        self.height = height
        self.color = color
        self.hover_color = hover_color
        self.background_rect = pg.Rect(x-3, y-3, width+6, height+6, border_radius=8)
        self.rect = pg.Rect(x, y, width, height, border_radius=8)
        self.font = pg.font.SysFont(INPUT_FONT, 12)
        self.rendered_text = self.font.render(text, True, BLACK) # Black text

    def draw_start_button(self, screen):

        # Change color on hover
        mouse_pos = pg.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            pg.draw.rect(screen, DARK_GRAY, (START_BUTTON_PLACEMENT_X-3, START_BUTTON_PLACEMENT_Y-3, START_BUTTON_WIDTH+6, START_BUTTON_HEIGHT+6), border_radius=16)
            pg.draw.rect(screen, self.hover_color, (START_BUTTON_PLACEMENT_X, START_BUTTON_PLACEMENT_Y, START_BUTTON_WIDTH, START_BUTTON_HEIGHT), border_radius=16)
        else:
            pg.draw.rect(screen, DARK_GRAY, (START_BUTTON_PLACEMENT_X-3, START_BUTTON_PLACEMENT_Y-3, START_BUTTON_WIDTH+6, START_BUTTON_HEIGHT+6), border_radius=16)
            pg.draw.rect(screen, self.color, (START_BUTTON_PLACEMENT_X, START_BUTTON_PLACEMENT_Y, START_BUTTON_WIDTH, START_BUTTON_HEIGHT), border_radius=16)
        
        # Center text on button
        text_rect = self.rendered_text.get_rect(center=self.rect.center)
        screen.blit(self.rendered_text, text_rect)

    def draw_log_button(self, screen):

        # Change color on hover
        mouse_pos = pg.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            pg.draw.rect(screen, DARK_GRAY, (LOG_BUTTON_PLACEMENT_X-3, LOG_BUTTON_PLACEMENT_Y-3, LOG_BUTTON_WIDTH+6, LOG_BUTTON_HEIGHT+6), border_radius=16)
            pg.draw.rect(screen, self.hover_color, (LOG_BUTTON_PLACEMENT_X, LOG_BUTTON_PLACEMENT_Y, LOG_BUTTON_WIDTH, LOG_BUTTON_HEIGHT), border_radius=16)
        else:
            pg.draw.rect(screen, DARK_GRAY, (LOG_BUTTON_PLACEMENT_X-3, LOG_BUTTON_PLACEMENT_Y-3, LOG_BUTTON_WIDTH+6, LOG_BUTTON_HEIGHT+6), border_radius=16)
            pg.draw.rect(screen, self.color, (LOG_BUTTON_PLACEMENT_X, LOG_BUTTON_PLACEMENT_Y, LOG_BUTTON_WIDTH, LOG_BUTTON_HEIGHT), border_radius=16)
        
        # Center text on button
        text_rect = self.rendered_text.get_rect(center=self.rect.center)
        screen.blit(self.rendered_text, text_rect)

    def draw_new_round_button(self, screen):

        # Change color on hover
        mouse_pos = pg.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            pg.draw.rect(screen, DARK_GRAY, (NEW_ROUND_BUTTON_PLACEMENT_X-3, NEW_ROUND_BUTTON_PLACEMENT_Y-3, NEW_ROUND_BUTTON_WIDTH+6, NEW_ROUND_BUTTON_HEIGHT+6), border_radius=16)
            pg.draw.rect(screen, self.hover_color, (NEW_ROUND_BUTTON_PLACEMENT_X, NEW_ROUND_BUTTON_PLACEMENT_Y, NEW_ROUND_BUTTON_WIDTH, NEW_ROUND_BUTTON_HEIGHT), border_radius=16)
        else:
            pg.draw.rect(screen, DARK_GRAY, (NEW_ROUND_BUTTON_PLACEMENT_X-3, NEW_ROUND_BUTTON_PLACEMENT_Y-3, NEW_ROUND_BUTTON_WIDTH+6, NEW_ROUND_BUTTON_HEIGHT+6), border_radius=16)
            pg.draw.rect(screen, self.color, (NEW_ROUND_BUTTON_PLACEMENT_X, NEW_ROUND_BUTTON_PLACEMENT_Y, NEW_ROUND_BUTTON_WIDTH, NEW_ROUND_BUTTON_HEIGHT), border_radius=16)
        
        # Center text on button
        text_rect = self.rendered_text.get_rect(center=self.rect.center)
        screen.blit(self.rendered_text, text_rect)

    def draw_num_player_button(self, screen):
        # Change color on hover
        mouse_pos = pg.mouse.get_pos()

        background_radius = int(self.background_rect.width/2)
        radius = int(self.rect.width/2)

        if self.rect.collidepoint(mouse_pos):
            pg.draw.circle(screen, DARK_GRAY, (self.background_rect.x + background_radius, self.background_rect.y + background_radius), background_radius)
            pg.draw.circle(screen, self.hover_color, (self.rect.x + radius, self.rect.y + radius), radius)
        else:
            pg.draw.circle(screen, DARK_GRAY, (self.background_rect.x + background_radius, self.background_rect.y + background_radius), background_radius)
            pg.draw.circle(screen, self.color, (self.rect.x + radius, self.rect.y + radius), radius)

        # Center text on button
        text_rect = self.rendered_text.get_rect(center=self.rect.center)
        screen.blit(self.rendered_text, text_rect)

    def draw_hud_button(self, screen):

        # Change color on hover
        mouse_pos = pg.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            pg.draw.rect(screen, DARK_GRAY, (self.background_rect.x, self.background_rect.y, self.background_rect.width, HUD_BUTTON_HEIGHT), border_radius=8)
            pg.draw.rect(screen, self.hover_color, (self.rect.x, self.rect.y, self.rect.width, HUD_BUTTON_HEIGHT), border_radius=8)
        else:
            pg.draw.rect(screen, DARK_GRAY, (self.background_rect.x, self.background_rect.y, self.background_rect.width, HUD_BUTTON_HEIGHT), border_radius=8)
            pg.draw.rect(screen, self.color, (self.rect.x, self.rect.y, self.rect.width, HUD_BUTTON_HEIGHT), border_radius=8)

        # Center text on button
        text_rect = self.rendered_text.get_rect(center=self.rect.center)
        screen.blit(self.rendered_text, text_rect)
        
    def handle_num_player_button_event(self, event, hold_num_players):

        if event.type == pg.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos

            if ((self.rect.x <= click_x <= self.rect.x+self.rect.width) and (self.rect.y <= click_y <=self.rect.y+self.rect.height)):

                self.active = not self.active
                if self.active:
                    self.color = DARK_GREEN

                    hold_num_players = int(self.text)

                return hold_num_players
            else:
                self.active = False
                if not self.active:
                    self.color = RED

                return hold_num_players

    def handle_start_button_event(self, event):

        if event.type == pg.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos

            if ((self.rect.x <= click_x <= self.rect.x+self.rect.width) and (self.rect.y <= click_y <=self.rect.y+self.rect.height)):
                self.active = not self.active
                if self.active:
                    self.color = DARK_GREEN
    
            else:
                self.active = False
                if not self.active:
                    self.color = RED

    def handle_log_button_event(self, event):

        if event.type == pg.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos

            if ((self.rect.x <= click_x <= self.rect.x+self.rect.width) and (self.rect.y <= click_y <=self.rect.y+self.rect.height)):
                self.active = True
                if self.active:
                    self.color = DARK_GREEN
    
            else:
                self.active = False
                if not self.active:
                    self.color = RED

        if event.type == pg.KEYDOWN:
            if (event.key == pg.K_l):
                self.active = True
                if self.active:
                    self.color = DARK_GREEN
    
            else:
                self.active = False
                if not self.active:
                    self.color = RED

    def handle_new_round_button_event(self, event):

        self.active = False

        if event.type == pg.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos

            if ((self.rect.x <= click_x <= self.rect.x+self.rect.width) and (self.rect.y <= click_y <=self.rect.y+self.rect.height)):
                self.active = True
                if self.active:
                    self.color = DARK_GREEN

        elif event.type == pg.KEYDOWN:

            if event.key == pg.K_s:
                self.active = True
                if self.active:
                    self.color = DARK_GREEN
        else:
            self.active = False
            if not self.active:
                self.color = RED

    def handle_hud_button(self, event):

        bet_decision = ""

        #Check for click on hud button
        if event.type == pg.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos
            
            if ((self.rect.x <= click_x <= self.rect.x+self.rect.width) and (self.rect.y <= click_y <=self.rect.y+self.rect.height)):
                self.active = True
                if self.active:
                    self.color = DARK_GREEN
                    bet_decision = self.text
                    if self.text == "RAISE":
                        self.rendered_text = self.font.render("", True, BLACK)
                        if self.rect.width <= HUD_BUTTON_WIDTH:
                            self.background_rect.width = ((self.background_rect.width-6) * 3) +3
                            self.rect.width = self.rect.width * 3
            else:
                self.active = False
                if not self.active:
                    self.color = RED
                    bet_decision = ""
                    if self.text == "RAISE":
                        self.rendered_text = self.font.render(self.text, True, BLACK)
                        self.background_rect.width = HUD_BUTTON_WIDTH + 6
                        self.rect.width = HUD_BUTTON_WIDTH
                        bet_decision = ""

        #Check for key down
        if event.type == pg.KEYDOWN:

            raise_keys = [pg.K_BACKSPACE,pg.K_RETURN,pg.K_0,pg.K_1,pg.K_2,pg.K_3,pg.K_4,pg.K_5,pg.K_6,pg.K_7,pg.K_8,pg.K_9,pg.K_r]

            if event.key == pg.K_f and self.text == "FOLD":

                self.active = not self.active

                if self.active:
                    self.color = DARK_GREEN
                    bet_decision = self.text
                else:
                    self.color = RED


            elif event.key == pg.K_c and self.text == "CHECK":

                self.active = not self.active
                if self.active:
                    self.color = DARK_GREEN
                    bet_decision = self.text
                else:
                    self.color = RED

            elif event.key == pg.K_r and self.text == "RAISE":

                self.active = True
                if self.active:
                    self.color = DARK_GREEN
                    bet_decision = self.text
                    self.rendered_text = self.font.render("", True, BLACK)
                    if self.rect.width <= HUD_BUTTON_WIDTH:
                        self.background_rect.width = ((self.background_rect.width-6) * 3) +3
                        self.rect.width = self.rect.width * 3
                else:
                    self.color = RED
                    self.rendered_text = self.font.render(self.text, True, BLACK)
                    self.background_rect.width = HUD_BUTTON_WIDTH + 6
                    self.rect.width = HUD_BUTTON_WIDTH
                    

            elif event.key in raise_keys:
                if self.text != "RAISE":
                    self.active = False
                    if not self.active:
                        self.color = RED

            else:
                self.active = False
                if not self.active:
                    self.color = RED
                    self.rendered_text = self.font.render(self.text, True, BLACK)
                    self.background_rect.width = HUD_BUTTON_WIDTH + 6
                    self.rect.width = HUD_BUTTON_WIDTH
     
        return bet_decision

def draw_card_images(CARDS_FOLDER_PATH):

        card_images = {}

        for filename in os.listdir(CARDS_FOLDER_PATH):
            try:
                # Use convert_alpha() for images with per-pixel transparency (like PNG)
                image = CARDS_FOLDER_PATH + filename
                image_surface = pg.image.load(image).convert_alpha()

                # width 55
                # height 80
                new_size = (CARD_WIDTH, CARD_HEIGHT)

                # Scale the image
                scaled_image = pg.transform.scale(image_surface, new_size) #new_size

                # Get the rectangle of the scaled image for positioning
                scaled_rect = scaled_image.get_rect()
                scaled_rect.center = (500, 800)

                card_images[filename] = scaled_image

            except pg.error as e:
                print(f"Error loading image: {e}")
                # Handle the error, maybe use a default image or exit
                pg.quit()
                exit()

        return card_images
        
def draw_chip_images(CHIPS_FOLDER_PATH):

    chip_images = {}

    for filename in os.listdir(CHIPS_FOLDER_PATH):
        try:
            # Use convert_alpha() for images with per-pixel transparency (like PNG)
            image = CHIPS_FOLDER_PATH + filename
            image_surface = pg.image.load(image).convert_alpha()

            new_size = ()

            if "TOP" in filename:
                new_size = (TOP_CHIP_WIDTH, TOP_CHIP_HEIGHT)

            if "FLAT" in filename:
                new_size = (FLAT_CHIP_WIDTH, FLAT_CHIP_HEIGHT)

            # Scale the image
            scaled_image = pg.transform.scale(image_surface, new_size) #new_size

            # Get the rectangle of the scaled image for positioning
            scaled_rect = scaled_image.get_rect()
            scaled_rect.center = (500, 800)

            chip_images[filename] = scaled_image

        except pg.error as e:
            print(f"Error loading image: {e}")
            # Handle the error, maybe use a default image or exit
            pg.quit()
            exit()

    return chip_images

def draw_pot(pot, chip_images):

    pot_image = []

    hold_pot = pot

    for chip_value in CHIP_VALUES:

        chip_image_stack = []

        divisible = int(hold_pot / chip_value)

        if divisible > 0:
            filename = f"{chip_value}FLAT.png"
            count = hold_pot / chip_value 
            count = int(count)
            chip_image_stack.append((chip_value, count, chip_images[filename]))
            hold_pot -= chip_value * count
            pot_image.append(chip_image_stack)

        if hold_pot == 0:
            break

    return pot_image

def draw_player_images(table, card_images, chip_images):

    player_images = []

    #range 2/3 of circles circumference angle
    start = (2*math.pi) / 6
    range = (2*math.pi) / 3
    range = range * 2 
    
    #break range in even slices for players
    player_location_angle = range / (table.num_players-1)
    start = start - math.pi / 2

    a = ELLIPSE_PLACEMENT_WIDTH / 2 # Semi-major axis (half the longest diameter)
    b = ELLIPSE_PLACEMENT_HEIGHT / 2 # Semi-minor axis (half the shortest diameter)

    player_spot = 0

    for player in table.players:

        player_image = {}
        
        player_image["ID"] = player.id
        player_image["CASH"] = player.cash
        player_image["BET"] = player.bet
        player_image["PLAYING"] = player.playing
        player_image["HAND"] = []

        if len(player.hand) == 2:
            player_image["HAND"] = []
            for card in player.hand:
                if not player.folded:
                    filename = f"{card.id}.png"
                    player_image["HAND"].append((card.id, card_images[filename]))
                elif player.folded == True:
                    player_image["HAND"].append(("Back Red 1.png", card_images["Back Red 1.png"]))
        if len(player.hand) == 0:
            player_image["HAND"] = []

        player_image["ORDER"] = None

        if table.small_blind == player.id:
            player_image["ORDER"] = chip_images["SMALLBLINDTOP.png"]
        elif table.big_blind == player.id:
            player_image["ORDER"] = chip_images["BIGBLINDTOP.png"]

        player_image["FOLDED"] = player.folded
        player_image["ALL_IN"] = player.all_in

        player_placement_x = int(ELLIPSE_CENTER_X + (a * math.cos(start + player_location_angle*player_spot)))
        player_placement_y = int(ELLIPSE_CENTER_Y + (b * math.sin(start + player_location_angle*player_spot)))

        player_image["PLAYER_PLACEMENT_X"] = player_placement_x
        player_image["PLAYER_PLACEMENT_Y"] = player_placement_y

        player_image["HUD_RECT"] = pg.Rect(player_placement_x, player_placement_x, PLAYER_WIDTH, PLAYER_HEIGHT, radius=PLAYER_RADIUS)
        player_image["ANGLE"] = start + player_location_angle*player_spot

        player_spot += 1

        player_images.append(player_image)
     
    return player_images

def draw_input_buttons():

    input_buttons = []

    for i in range(MAX_PLAYERS-1):
        button_lable = i+2
        input_buttons.append(
            Button(f"{button_lable}", NUM_PLAYERS_BUTTON_X + i*2*NUM_PLAYERS_BUTTON_WIDTH, 
                        NUM_PLAYERS_BUTTON_Y, NUM_PLAYERS_BUTTON_WIDTH, 
                        NUM_PLAYERS_BUTTON_HEIGHT, RED, GREEN))

    return input_buttons

def draw_start_button():

    start_button = Button(
        "START GAME", 
        START_BUTTON_PLACEMENT_X, START_BUTTON_PLACEMENT_Y, 
        START_BUTTON_WIDTH, START_BUTTON_HEIGHT, 
        RED, GREEN)

    return start_button

def draw_log_button():

    log_button = Button(
        "LOG ACTION",
        LOG_BUTTON_PLACEMENT_X, LOG_BUTTON_PLACEMENT_Y,
        LOG_BUTTON_WIDTH, LOG_BUTTON_HEIGHT,
        RED, GREEN
    )

    return log_button

def draw_new_round_button():
    new_round_button = Button(
        "START ROUND",
        NEW_ROUND_BUTTON_PLACEMENT_X, NEW_ROUND_BUTTON_PLACEMENT_Y,
        NEW_ROUND_BUTTON_WIDTH, NEW_ROUND_BUTTON_HEIGHT,
        RED, GREEN
    )

    return new_round_button

def draw_hud_buttons():

    hud_buttons = {}

    hud_buttons[f"FOLD"] = Button("FOLD", HUD_PLAYERS_BUTTON_X, HUD_PLAYERS_BUTTON_Y, HUD_BUTTON_WIDTH, HUD_BUTTON_HEIGHT, RED, GREEN)
    hud_buttons[f"CHECK"] = Button("CHECK", HUD_PLAYERS_BUTTON_X+(1.5*HUD_BUTTON_WIDTH), HUD_PLAYERS_BUTTON_Y, HUD_BUTTON_WIDTH, HUD_BUTTON_HEIGHT, RED, GREEN)
    hud_buttons[f"RAISE"] = Button("RAISE", HUD_PLAYERS_BUTTON_X+(3*HUD_BUTTON_WIDTH), HUD_PLAYERS_BUTTON_Y, HUD_BUTTON_WIDTH, HUD_BUTTON_HEIGHT, RED, GREEN)

    return hud_buttons

def draw_input_boxes(input_font):

    bet_input_box = InputBox(INPUT_WINDOW_PLACEMENT_X+INPUT_WINDOW_WIDTH/2+INPUT_BOX_WIDTH/4, INPUT_WINDOW_PLACEMENT_Y+195, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT, input_font)
    wallet_input_box = InputBox(INPUT_WINDOW_PLACEMENT_X+INPUT_WINDOW_WIDTH/2+INPUT_BOX_WIDTH/4, INPUT_WINDOW_PLACEMENT_Y+230, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT, input_font)

    return bet_input_box, wallet_input_box

def draw_raise_input_box(input_font):
    raise_input_box = InputBox(HUD_PLAYERS_BUTTON_X+(3.75*HUD_BUTTON_WIDTH), HUD_PLAYERS_BUTTON_Y+HUD_BUTTON_HEIGHT/3, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT, input_font)
    return raise_input_box

class PygameUI:
    
    def __init__(self, table: TableState):
        
        self.hand_counter = 1
        self.counter = 0
        self.table = table

        self.hold_num_players = 0
        self.hold_buy_in = 0
        self.hold_wallet = 0
        self.hold_river = []
        self.hold_player_images = []
        self.hold_winner_images = []
        self.hold_player_action = ""

        self.user_input = ""

        #pygame engine running
        self.running = True
        self.phase = GamePhase.SETUP

        #Table has been made 
        self.game_running = False

        #UI
        pg.init()

        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption("Poker Table")
        
        self.clock = pg.time.Clock()

        self.pot_font = pg.font.SysFont(POT_FONT, 20)
        self.money_font = pg.font.SysFont(MONEY_FONT, 10)
        self.player_font = pg.font.SysFont(PLAYER_FONT, 12)
        self.header_font = pg.font.SysFont(INPUT_FONT, 25)
        self.hud_header_font = pg.font.SysFont(INPUT_FONT, 20)
        self.input_font = pg.font.SysFont(INPUT_FONT, 10)

        #Background
        try:
            background_image = pg.image.load(BACKGROUND_IMAGE_PATH)
            background_image = background_image.convert()
            self.scaled_background_image = pg.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT)) 

            self.card_images = draw_card_images(CARDS_FOLDER_PATH)
            self.chip_images = draw_chip_images(CHIPS_FOLDER_PATH)

        except pg.error as e:
            print(f"Error loading image: {e}")
            sys.exit()

        #images

        #physical objects
        self.input_buttons = draw_input_buttons()
        self.bet_input_box, self.wallet_input_box = draw_input_boxes(self.input_font)
        self.start_button = draw_start_button()

        self.player_hud_buttons = draw_hud_buttons()
        self.log_button = draw_log_button()
        self.new_round_button = draw_new_round_button()
        self.raise_input_box = draw_raise_input_box(self.input_font)
        
        self.com_deck = []
        for i in range(5):
            self.com_deck.append(self.card_images["Back Red 1.png"])

    def draw_community_cards(self, cards):

        i = 0
        for card in cards:
            filename = card.id + ".png"
            self.com_deck[i] = (self.card_images[filename])
            i+=1

    # =========================
    # Event handling
    # =========================

    def _handle_setup_events(self, screen):
        for event in pg.event.get():
            #buy in
            self.bet_input_box.handle_event(event)
            if self.bet_input_box.stored_input != 0:
                self.hold_buy_in = self.bet_input_box.stored_input
            
            #wallet input box
            self.wallet_input_box.handle_event(event)
            if self.wallet_input_box.stored_input != 0:
                self.hold_wallet = self.wallet_input_box.stored_input

            # loop through player number buttons
            for button in self.input_buttons:
                hold = button.handle_num_player_button_event(event, self.hold_num_players)
                if hold in [1,2,3,4,5,6]:
                    self.hold_num_players = hold

            self.start_button.handle_start_button_event(event)
            if self.start_button.active:
                self.game_running = True
                table = TableState(self.hold_num_players, self.hold_buy_in, self.hold_wallet)
                self.table = table
            else:
                self.game_running = False

            if event.type == pg.QUIT:
                self.running = False 
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                else:
                    # Append the Unicode character of the pressed key
                    self.user_input += event.unicode
            elif event.type == pg.MOUSEBUTTONDOWN: # Handle mouse clicks
                # Check if the user clicked on the input_box rect
                pos = event.pos

    def _handle_game_events(self, screen, hc):
        
        for event in pg.event.get():
                
            # handle event for all hud buttons
            f = self.player_hud_buttons["FOLD"].handle_hud_button(event)
            if f != "":
                self.hold_player_action=""
                self.hold_player_action = f
            c = self.player_hud_buttons["CHECK"].handle_hud_button(event)
            if c != "":
                self.hold_player_action=""
                self.hold_player_action = c
            r = self.player_hud_buttons["RAISE"].handle_hud_button(event)
            if self.player_hud_buttons["RAISE"].active: self.raise_input_box.active = True
            if r != "":
                self.hold_player_action=""
                self.hold_player_action = r

            # handle event for raise input
            self.raise_input_box.handle_event(event)

            # handle event for log button and log player action
            self.log_button.handle_log_button_event(event)
            
            if self.log_button.active:

                if self.hold_player_action == "RAISE" and self.raise_input_box.stored_input >= hc.table.current_bet*2:
                    self._send_action(hc, self.hold_player_action, self.raise_input_box.stored_input)
                    self.hold_player_action = ""
                    self.raise_input_box.text = ""
                    self.raise_input_box.stored_input = 0
                     
                elif self.hold_player_action == "CHECK":
                    self._send_action(hc, "CALL", 0)
                    self.hold_player_action = ""
                    
                elif self.hold_player_action == "FOLD":
                    self._send_action(hc, self.hold_player_action, 0)
                    self.hold_player_action = ""

            if hc.phase == GamePhase.SHOWDOWN:
                self.new_round_button.handle_new_round_button_event(event)
                if self.new_round_button.active:
                    self.hand_counter+=1
                    self.new_round_button.active = False
                    hc._advance_phase()
            
            if event.type == pg.QUIT:
                self.running = False 
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                else:
                    # Append the Unicode character of the pressed key
                    self.user_input += event.unicode
            elif event.type == pg.MOUSEBUTTONDOWN: # Handle mouse clicks
                # Check if the user clicked on the input_box rect
                pos = event.pos


    def _send_action(self, hc, action_type, amount):

        player_id = hc.betting_round.current_index

        action_t = ActionType.CHECK

        if action_type == "RAISE":
            action_t = ActionType.RAISE
        elif action_type == "CALL":
            action_t = ActionType.CALL
        elif action_type == "CHECK":
            action_t = ActionType.CHECK
        elif action_type == "FOLD":
            action_t = ActionType.FOLD

        action = PlayerAction(
            action_type=action_t,
            player_index=int(player_id),
            raise_amount=amount
        )

        hc.apply_action(action)
        return action

    # =========================
    # Rendering
    # =========================

    #Draw setup screens
    def _draw_setup(self, screen):

        # Input Window
        pg.draw.rect(screen, DARK_GRAY, (INPUT_WINDOW_PLACEMENT_X-5, INPUT_WINDOW_PLACEMENT_Y-5, INPUT_WINDOW_WIDTH+10, INPUT_WINDOW_HEIGHT+10), border_radius = PLAYER_RADIUS) 
        pg.draw.rect(screen, GRAY, (INPUT_WINDOW_PLACEMENT_X, INPUT_WINDOW_PLACEMENT_Y, INPUT_WINDOW_WIDTH, INPUT_WINDOW_HEIGHT), border_radius = PLAYER_RADIUS)  
        
        input_window_header_text = self.header_font.render("GAME INPUT WINDOW", True, WHITE)

        input_num_players_text = self.player_font.render(f"SELECT YOUR DESIRED NUMBER OF PLAYERS: {self.hold_num_players}", True, BLACK)
        input_buy_in_text = self.player_font.render("INPUT DESIRED BUY-IN: ", True, BLACK)
        input_wallet_size_text = self.player_font.render("INPUT DESIRED WALLET: ", True, BLACK)

        screen.blit(input_window_header_text, (INPUT_WINDOW_PLACEMENT_X+20, INPUT_WINDOW_PLACEMENT_Y+60))
        screen.blit(input_num_players_text, (INPUT_WINDOW_PLACEMENT_X+20, INPUT_WINDOW_PLACEMENT_Y+120))
        screen.blit(input_buy_in_text, (INPUT_WINDOW_PLACEMENT_X+20, INPUT_WINDOW_PLACEMENT_Y+195))
        screen.blit(input_wallet_size_text, (INPUT_WINDOW_PLACEMENT_X+20, INPUT_WINDOW_PLACEMENT_Y+230))

        for button in self.input_buttons:
            button.draw_num_player_button(screen)

        self.bet_input_box.draw(screen)
        self.wallet_input_box.draw(screen)

        if (self.hold_num_players>0) and (self.hold_buy_in>0) and (self.hold_wallet>0):
            self.start_button.draw_start_button(screen)


    # Draw in game screens
    def _draw_phase(self, screen, hc):

        if hc.phase == GamePhase.GAMEOVER:
            self.game_running=False

        phase_text = ""
        transparent_surface = pg.Surface((CARD_WIDTH-2, CARD_HEIGHT-2), pg.SRCALPHA)
        transparent_surface.fill(TRANSPARENT_GOLD)

        if hc.phase == GamePhase.PREFLOP: 
            phase_text = "PRE FLOP"
        if hc.phase == GamePhase.FLOP: 
            phase_text = "FLOP"
        if hc.phase == GamePhase.TURN: 
            phase_text = "TURN"
        if hc.phase == GamePhase.RIVER: 
            phase_text = "RIVER"
        if hc.phase == GamePhase.SHOWDOWN: 
            phase_text = "SHOWDOWN"

        screen.blit(self.scaled_background_image, (0,0))

        # Dealer station
        screen.blit(self.chip_images["DEALERTOP.png"], (DEALER_PLACEMENT_X-2*TOP_CHIP_WIDTH, DEALER_PLACEMENT_Y))
        deck_spread=1
        for card in hc.deck.cards: 
            screen.blit(self.card_images["Back Red 1.png"], (DEALER_PLACEMENT_X, DEALER_PLACEMENT_Y-1*deck_spread))
            deck_spread+=1

        self.pot_image = draw_pot(hc.table.pot, self.chip_images)
        self.player_images = draw_player_images(hc.table, self.card_images, self.chip_images)

        # Community cards
        if len(hc.table.community_cards)>0:
            self.draw_community_cards(hc.table.community_cards)
            if len(self.com_deck) == 5:
                self.hold_river = self.com_deck
                self.hold_player_images = self.player_images.copy()

        if hc.phase == GamePhase.SHOWDOWN:
            self.draw_community_cards(hc.final_community_cards)
            self.hold_river = self.com_deck.copy()
            if len(hc.winning_players) > 0:
                self.hold_winner_images = []
                for player_image in self.hold_player_images:
                    won = False
                    for player in hc.winning_players:
                        if player.id == player_image["ID"]:
                            won = True
                    if won:
                        self.hold_winner_images.append(player_image)
                
        if hc.phase == GamePhase.FLOP: 
            screen.blit(self.com_deck[0], (COMMUNITY_CARDS_PLACEMENT_X, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[1], (COMMUNITY_CARDS_PLACEMENT_X+CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[2], (COMMUNITY_CARDS_PLACEMENT_X+2*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
        elif hc.phase == GamePhase.TURN: 
            screen.blit(self.com_deck[0], (COMMUNITY_CARDS_PLACEMENT_X, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[1], (COMMUNITY_CARDS_PLACEMENT_X+CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[2], (COMMUNITY_CARDS_PLACEMENT_X+2*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[3], (COMMUNITY_CARDS_PLACEMENT_X+3*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
        elif hc.phase == GamePhase.RIVER: 
            screen.blit(self.com_deck[0], (COMMUNITY_CARDS_PLACEMENT_X, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[1], (COMMUNITY_CARDS_PLACEMENT_X+CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[2], (COMMUNITY_CARDS_PLACEMENT_X+2*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[3], (COMMUNITY_CARDS_PLACEMENT_X+3*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
            screen.blit(self.com_deck[4], (COMMUNITY_CARDS_PLACEMENT_X+4*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
        elif hc.phase == GamePhase.SHOWDOWN:
            try:
                screen.blit(self.hold_river[0], (COMMUNITY_CARDS_PLACEMENT_X, COMMUNITY_CARDS_PLACEMENT_Y))
                for card in hc.best_five_card_combo:
                    card_image = self.card_images[f"{card.id}.png"]
                    if self.hold_river[0] == card_image:
                        screen.blit(transparent_surface, ((COMMUNITY_CARDS_PLACEMENT_X+1), COMMUNITY_CARDS_PLACEMENT_Y+1))
                screen.blit(self.hold_river[1], (COMMUNITY_CARDS_PLACEMENT_X+CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
                for card in hc.best_five_card_combo:
                    card_image = self.card_images[f"{card.id}.png"]
                    if self.hold_river[1] == card_image:
                        screen.blit(transparent_surface, ((COMMUNITY_CARDS_PLACEMENT_X+1)+CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y+1))
                screen.blit(self.hold_river[2], (COMMUNITY_CARDS_PLACEMENT_X+2*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
                for card in hc.best_five_card_combo:
                    card_image = self.card_images[f"{card.id}.png"]
                    if self.hold_river[2] == card_image:
                        screen.blit(transparent_surface, ((COMMUNITY_CARDS_PLACEMENT_X+1)+2*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y+1))
                screen.blit(self.hold_river[3], (COMMUNITY_CARDS_PLACEMENT_X+3*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
                for card in hc.best_five_card_combo:
                    card_image = self.card_images[f"{card.id}.png"]
                    if self.hold_river[3] == card_image:
                        screen.blit(transparent_surface, ((COMMUNITY_CARDS_PLACEMENT_X+1)+3*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y+1))
                screen.blit(self.hold_river[4], (COMMUNITY_CARDS_PLACEMENT_X+4*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y))
                for card in hc.best_five_card_combo:
                    card_image = self.card_images[f"{card.id}.png"]
                    if self.hold_river[4] == card_image:
                        screen.blit(transparent_surface, ((COMMUNITY_CARDS_PLACEMENT_X+1)+4*CARD_WIDTH, COMMUNITY_CARDS_PLACEMENT_Y+1))
            except IndexError:
                print(f"IndexError length hold river {len(self.hold_river)}")

        # Draw Pot
        pot_width = 1
        if hc.table.pot > 0:
            for stack in self.pot_image:
                for i in range(stack[0][1]):
                    height_buffer = FLAT_CHIP_HEIGHT/2
                    screen.blit(stack[0][2], (POT_PLACEMENT_X + pot_width * FLAT_CHIP_WIDTH, POT_PLACEMENT_Y - i* height_buffer))
                pot_width+=1

                pot_text = self.pot_font.render("$" + str(hc.table.pot), True, GOLD) # Anti-alias=True
                screen.blit(pot_text, (POT_PLACEMENT_X + 2*FLAT_CHIP_WIDTH, POT_PLACEMENT_Y + 25))

        # Draw Players and player data
        for player_image in self.player_images:
            
            pg.draw.rect(screen, DARK_GRAY, (player_image["PLAYER_PLACEMENT_X"]-5, player_image["PLAYER_PLACEMENT_Y"]-5, PLAYER_WIDTH+10, PLAYER_HEIGHT+10), border_radius = PLAYER_RADIUS) # Player background

            if player_image["PLAYING"]:
                pg.draw.rect(screen, GRAY, (player_image["PLAYER_PLACEMENT_X"], player_image["PLAYER_PLACEMENT_Y"], PLAYER_WIDTH, PLAYER_HEIGHT), border_radius = PLAYER_RADIUS) # Player 
            elif not player_image["PLAYING"]:
                pg.draw.rect(screen, DARK_RED, (player_image["PLAYER_PLACEMENT_X"], player_image["PLAYER_PLACEMENT_Y"], PLAYER_WIDTH, PLAYER_HEIGHT), border_radius = PLAYER_RADIUS)

            wallet_text = self.money_font.render("WALLET: $" + str(player_image["CASH"]), True, GOLD)

            bet_text = ""

            if player_image["PLAYING"] == False:
                bet_text = self.money_font.render("out", True, BLACK)
            elif player_image["FOLDED"] == True:
                bet_text = self.money_font.render("fold", True, BLACK)
            elif player_image["ALL_IN"] == True:
                bet_text = self.money_font.render("ALL IN", True, GOLD)
            elif player_image["BET"] != hc.table.current_bet:
                bet_text = self.money_font.render("BET: $" + str(player_image["BET"]), True, RED)
            elif player_image["BET"] == hc.table.current_bet:
                bet_text = self.money_font.render("BET: $" + str(player_image["BET"]), True, GREEN)
    
            player_text = self.player_font.render("PLAYER " + str(player_image["ID"]), True, BLACK)

            screen.blit(player_text, (player_image["PLAYER_PLACEMENT_X"]+25, player_image["PLAYER_PLACEMENT_Y"]+25))
            screen.blit(wallet_text, (player_image["PLAYER_PLACEMENT_X"]+25, player_image["PLAYER_PLACEMENT_Y"]+50))
            screen.blit(bet_text, (player_image["PLAYER_PLACEMENT_X"]+25, player_image["PLAYER_PLACEMENT_Y"]+70))

            if player_image["ORDER"]:
                screen.blit(player_image["ORDER"], (player_image["PLAYER_PLACEMENT_X"]+90, player_image["PLAYER_PLACEMENT_Y"]+15))

            # DRAW HAND 
            cards_to_center_x = 0
            cards_to_center_y = 100

            if player_image["PLAYER_PLACEMENT_X"] > 500:
                cards_to_center_x -= CARD_WIDTH/2
            if player_image["PLAYER_PLACEMENT_X"] < 500:
                cards_to_center_x = CARD_WIDTH
            if player_image["PLAYER_PLACEMENT_Y"] > 440:
                cards_to_center_y -= CARD_HEIGHT
                if player_image["PLAYER_PLACEMENT_X"] > 500:
                    cards_to_center_x -= CARD_WIDTH
                if player_image["PLAYER_PLACEMENT_X"] < 500:
                    cards_to_center_x += CARD_WIDTH*.75

            if hc.phase != GamePhase.SHOWDOWN:

                if player_image["PLAYING"]:
                    if player_image["ID"] == hc.betting_round.current_index:
                        screen.blit(player_image["HAND"][1][1], (player_image["PLAYER_PLACEMENT_X"]+cards_to_center_x, player_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))
                        screen.blit(player_image["HAND"][0][1], (player_image["PLAYER_PLACEMENT_X"]+40+cards_to_center_x, player_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))
                    else:
                        screen.blit(self.card_images["Back Red 1.png"], (player_image["PLAYER_PLACEMENT_X"]+cards_to_center_x, player_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))
                        screen.blit(self.card_images["Back Red 1.png"], (player_image["PLAYER_PLACEMENT_X"]+40+cards_to_center_x, player_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))

            elif hc.phase == GamePhase.SHOWDOWN:

                for winner_image in self.hold_winner_images:

                    if (player_image["ID"] == winner_image["ID"]):

                        screen.blit(winner_image["HAND"][1][1], (winner_image["PLAYER_PLACEMENT_X"]+cards_to_center_x, winner_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))
                        
                        for card in hc.best_five_card_combo:
                            card_image = self.card_images[f"{card.id}.png"]
                            if card_image == winner_image["HAND"][1][1]:
                                screen.blit(transparent_surface, (winner_image["PLAYER_PLACEMENT_X"]+1+cards_to_center_x, winner_image["PLAYER_PLACEMENT_Y"]+1+cards_to_center_y))

                        screen.blit(winner_image["HAND"][0][1], (winner_image["PLAYER_PLACEMENT_X"]+40+cards_to_center_x, winner_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))

                        for card in hc.best_five_card_combo:
                            card_image = self.card_images[f"{card.id}.png"]
                            if card_image == winner_image["HAND"][0][1]:
                                screen.blit(transparent_surface, (winner_image["PLAYER_PLACEMENT_X"]+1+40+cards_to_center_x, winner_image["PLAYER_PLACEMENT_Y"]+1+cards_to_center_y))

                    else:
                        if player_image["PLAYING"]:
                            screen.blit(self.card_images["Back Red 1.png"], (player_image["PLAYER_PLACEMENT_X"]+cards_to_center_x, player_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))
                            screen.blit(self.card_images["Back Red 1.png"], (player_image["PLAYER_PLACEMENT_X"]+40+cards_to_center_x, player_image["PLAYER_PLACEMENT_Y"]+cards_to_center_y))

        # Player HUD windown 
        # Check player is current player or SHOWDOWN
        # Handle None type is pre betting round
        # When enough user information is provided to log a player action

        if hc.phase != GamePhase.SHOWDOWN: 

            hud_player_image = self.player_images[hc.betting_round.current_index]

            pg.draw.rect(screen, DARK_GRAY, (HUD_WINDOW_PLACEMENT_X-5, HUD_WINDOW_PLACEMENT_Y-5, HUD_WINDOW_WIDTH+10, HUD_WINDOW_HEIGHT+10), border_radius = PLAYER_RADIUS) 
            pg.draw.rect(screen, GRAY, (HUD_WINDOW_PLACEMENT_X, HUD_WINDOW_PLACEMENT_Y, HUD_WINDOW_WIDTH, HUD_WINDOW_HEIGHT), border_radius = PLAYER_RADIUS)  
            hud_window_header_text = self.hud_header_font.render(f"HAND {self.hand_counter} {phase_text}: PLAYER {hud_player_image["ID"]}", True, WHITE)
            bet_text = self.player_font.render(f"CURRENT BET: ${float(hc.table.current_bet)}", True, BLACK)
            wallet_size_text = self.player_font.render(f"CURRENT WALLET: ${float(hud_player_image["CASH"])}", True, BLACK)
            screen.blit(hud_window_header_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y+110))
            screen.blit(bet_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y+225))
            screen.blit(wallet_size_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y+260))

            self.player_hud_buttons["FOLD"].draw_hud_button(screen)
            self.player_hud_buttons["CHECK"].draw_hud_button(screen)
            self.player_hud_buttons["RAISE"].draw_hud_button(screen)
            
            if self.player_hud_buttons["RAISE"].active:
                self.raise_input_box.draw(screen)

            if self.player_hud_buttons["FOLD"].active:
                self.log_button.draw_log_button(screen)
            elif self.player_hud_buttons["CHECK"].active:
                self.log_button.draw_log_button(screen)
            elif (self.player_hud_buttons["RAISE"].active) & (self.raise_input_box.stored_input != 0):
                self.log_button.draw_log_button(screen)

        elif hc.phase == GamePhase.SHOWDOWN:

            pg.draw.rect(screen, DARK_GRAY, (HUD_WINDOW_PLACEMENT_X-5, HUD_WINDOW_PLACEMENT_Y-5, HUD_WINDOW_WIDTH+10, HUD_WINDOW_HEIGHT+10), border_radius = PLAYER_RADIUS) 
            pg.draw.rect(screen, GRAY, (HUD_WINDOW_PLACEMENT_X, HUD_WINDOW_PLACEMENT_Y, HUD_WINDOW_WIDTH, HUD_WINDOW_HEIGHT), border_radius = PLAYER_RADIUS)  

            #get_winning_five_card
            increment = 0
            for card in hc.best_five_card_combo:
                card_image = self.card_images[f"{card.id}.png"]

                rect_x = (HUD_WINDOW_PLACEMENT_X+HUD_WINDOW_WIDTH/2) + increment * (CARD_WIDTH/3)
                rect_y = (HUD_WINDOW_PLACEMENT_Y+HUD_WINDOW_HEIGHT-HUD_WINDOW_HEIGHT/2 + CARD_HEIGHT/4)

                new_size = (CARD_WIDTH/2, CARD_HEIGHT/2)

                # Scale the image
                scaled_image = pg.transform.scale(card_image, new_size) #new_size

                screen.blit(scaled_image, (rect_x, rect_y))
                increment+=1

            winning_players = "WINNERS: "
            for winner in hc.winning_players:
                winning_players = winning_players + f"PLAYER {winner.id} "

            hud_window_header_text = self.hud_header_font.render(f"HAND {self.hand_counter}: {phase_text}", True, WHITE)
            winning_players_text = self.player_font.render(winning_players, True, BLACK)
            pot_share_text = self.player_font.render(f"POT SHARE: ${hc.pot_share}", True, BLACK)
            best_hand_name_text = self.player_font.render(f"HAND: {hc.best_hand_name}", True, BLACK)
            screen.blit(hud_window_header_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y+110))
            screen.blit(winning_players_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y + HUD_WINDOW_HEIGHT/2))
            screen.blit(pot_share_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y+225))
            screen.blit(best_hand_name_text, (HUD_WINDOW_PLACEMENT_X+20, HUD_WINDOW_PLACEMENT_Y+260))

            self.new_round_button.draw_new_round_button(screen)

        #Show Screen
        pg.display.flip()

    def _draw_game_over(self, screen, hc):

        screen.fill(BLACK)

        # End Game Window
        pg.draw.rect(screen, DARK_GRAY, (GAME_OVER_PLACEMENT_X-5, GAME_OVER_PLACEMENT_Y-5, GAME_OVER_WIDTH+10, GAME_OVER_HEIGHT+10), border_radius = PLAYER_RADIUS) 
        pg.draw.rect(screen, GRAY, (GAME_OVER_PLACEMENT_X, GAME_OVER_PLACEMENT_Y, GAME_OVER_WIDTH, GAME_OVER_HEIGHT), border_radius = PLAYER_RADIUS)
        
        winner = hc.winning_players[0]

        cash = f"${winner.cash:,.2f}"
        pot_share = f"${winner.largest_potshare:,.2f}"

        header_text = self.header_font.render(f"GAME OVER", True, WHITE)
        header_winnner_text = self.header_font.render(f"WINNER: PLAYER {winner.id}", True, WHITE)
        number_of_hands_text = self.header_font.render(f"TOTAL NUMBER OF HANDS: {self.hand_counter-1}", True, WHITE)
        best_five_card_text = self.header_font.render("BEST FIVE CARD HAND", True, WHITE) 
        
        screen.blit(header_text, (GAME_OVER_PLACEMENT_X+20, GAME_OVER_PLACEMENT_Y+40))
        screen.blit(header_winnner_text, (GAME_OVER_PLACEMENT_X+20, GAME_OVER_PLACEMENT_Y+70))
        screen.blit(number_of_hands_text, (GAME_OVER_PLACEMENT_X+GAME_OVER_WIDTH/2, GAME_OVER_PLACEMENT_Y+40))
        screen.blit(best_five_card_text, (GAME_OVER_PLACEMENT_X+GAME_OVER_WIDTH/2, GAME_OVER_PLACEMENT_Y+70))

        # Best hand image
        increment = 0
        for card in winner.best_hand:
            card_image = self.card_images[f"{card.id}.png"]
            rect_x = (GAME_OVER_PLACEMENT_X+GAME_OVER_WIDTH/2) + increment * (CARD_WIDTH/2)
            rect_y = (GAME_OVER_PLACEMENT_Y+110)
            new_size = (CARD_WIDTH/2, CARD_HEIGHT/2)
            # Scale the image
            scaled_image = pg.transform.scale(card_image, new_size) #new_size
            screen.blit(scaled_image, (rect_x, rect_y))
            increment+=1

        cash_image = draw_pot(winner.cash, self.chip_images)
        pot_share_image = draw_pot(winner.largest_potshare, self.chip_images)
        
        # Draw Pot
        pot_width = 1
        
        # Draw Pot
        pot_width = 1
        for stack in cash_image:
            for i in range(stack[0][1]):
                height_buffer = FLAT_CHIP_HEIGHT/2
                screen.blit(stack[0][2], (GAME_OVER_PLACEMENT_X+20 + pot_width * FLAT_CHIP_WIDTH, GAME_OVER_PLACEMENT_Y+GAME_OVER_HEIGHT/2+5*FLAT_CHIP_HEIGHT - i* height_buffer))
            pot_width+=1

            pot_text = self.pot_font.render("WINNER CASH " + cash, True, GOLD) # Anti-alias=True
            screen.blit(pot_text, (GAME_OVER_PLACEMENT_X+20, INPUT_WINDOW_PLACEMENT_Y+100 + 25))

        for stack in pot_share_image:
            for i in range(stack[0][1]):
                height_buffer = FLAT_CHIP_HEIGHT/2
                screen.blit(stack[0][2], (GAME_OVER_PLACEMENT_X + GAME_OVER_WIDTH/2 + pot_width * FLAT_CHIP_WIDTH, GAME_OVER_PLACEMENT_Y+GAME_OVER_HEIGHT/2+5*FLAT_CHIP_HEIGHT - i* height_buffer))
            pot_width+=1

            pot_share_text = self.pot_font.render("LARGEST POTSHARE " + pot_share, True, GOLD) # Anti-alias=True
            screen.blit(pot_share_text, (GAME_OVER_PLACEMENT_X + GAME_OVER_WIDTH/2 + 2*FLAT_CHIP_WIDTH, INPUT_WINDOW_PLACEMENT_Y+100 + 25))

        # Draw axes
        pg.draw.line(screen, WHITE, (GRAPH_START_X, GRAPH_START_Y), (GRAPH_START_X, GRAPH_START_Y - GRAPH_HEIGHT), 2) # Y-axis
        pg.draw.line(screen, WHITE, (GRAPH_START_X, GRAPH_START_Y), (GRAPH_START_X + len(winner.cash_by_round) * X_SPACING, GRAPH_START_Y), 2) # X-axis

        # Draw the line graph
        if len(winner.cash_by_round) > 1:
            points = [self.get_screen_coord(i, dp, winner.cash_by_round) for i, dp in enumerate(winner.cash_by_round)]
            # Use the pygame.draw.lines function to connect multiple points
            pg.draw.lines(screen, GOLD, False, points, 3)

        pg.display.flip()

    def get_screen_coord(self, x_index, y_value, cash_by_round):
        # Invert the y-axis (Pygame y-origin is top-left) and scale
        screen_x = GRAPH_START_X + x_index * X_SPACING
        screen_y = GRAPH_START_Y - (y_value * (GRAPH_HEIGHT / max(cash_by_round)))
        return int(screen_x), int(screen_y)
    
    # ==========================
    # Toggle between setup and gameplay visuals
    # ==========================

    def _render(self, screen, hc):

        phase = GamePhase.SETUP

        if hc == None:
            phase = GamePhase.SETUP
        else:
            phase = hc.phase

        if phase == GamePhase.SETUP: # and not self.running
            self._draw_setup(screen)
        elif phase != GamePhase.GAMEOVER: # elif self.engine is not None
            self._draw_phase(screen, hc)
        elif phase == GamePhase.GAMEOVER:
            self._draw_game_over(screen, hc)

        pg.display.flip()

    # =========================
    # Main loop
    # =========================

    def run(self, screen):

        # renders setup window input
        while (not self.game_running) and self.running:

            self.clock.tick(60)
            self._handle_setup_events(screen)
            self._render(screen, None)

        hc = HandController(self.table, HandEvaluator())
        hc.start_hand()

        # renders game window input
        while self.game_running and self.running:
            
            self.clock.tick(60)
            self._handle_game_events(screen, hc)
            self._render(screen, hc)

        while self.running:
            self.clock.tick(60)
            self._handle_game_events(screen, hc)
            self._render(screen, hc)

        pg.quit()
