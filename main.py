import numpy as np
import random
import tkinter as tk
import time  # Pour mesurer le temps écoulé

# Définir le labyrinthe (0 = espace libre, 1 = mur)
maze = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
])

# Définir la position de départ et d'objectif
goal = (9, 9)  # Position de l'objectif
start = (0, 0)  # Position de départ

# Déplacements possibles (gauche, droite, haut, bas)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Taille du labyrinthe
num_rows, num_cols = maze.shape

# Paramètres du Q-learning
learning_rate = 0.7  # Taux d'apprentissage (à quel point le nouvel apprentissage remplace l'ancien)
discount_factor = 0.9  # Importance des récompenses futures
exploration_rate = 1.0  # Probabilité d'explorer au lieu d'exploiter
exploration_decay = 0.995  # Décroissance du taux d'exploration à chaque épisode
min_exploration_rate = 0.01  # Taux d'exploration minimal
num_episodes = 15000  # Nombre total d'épisodes d'entraînement

# Initialisation de la table Q avec des zéros
Q_table = np.zeros((num_rows, num_cols, len(actions)))

# Fonction pour vérifier si un déplacement est valide
def is_valid_move(row, col):
    return 0 <= row < num_rows and 0 <= col < num_cols and maze[row, col] == 0

# Fonction pour choisir une action selon une politique epsilon-greedy
def choose_action(state, exploration_rate):
    if random.uniform(0, 1) < exploration_rate:
        return random.randint(0, len(actions) - 1)  # Exploration : choisir une action aléatoire
    else:
        return np.argmax(Q_table[state])  # Exploitation : choisir la meilleure action connue

# Entraînement avec l'algorithme Q-learning
def train_qlearning():
    global exploration_rate
    for episode in range(num_episodes):
        state = start  # Réinitialiser à l'état de départ pour chaque épisode
        while state != goal:
            # Choisir une action
            action_index = choose_action(state, exploration_rate)
            action = actions[action_index]

            # Calculer le prochain état en fonction de l'action choisie
            next_state = (state[0] + action[0], state[1] + action[1])

            # Vérifier si le déplacement est valide
            if is_valid_move(*next_state):
                reward = 100 if next_state == goal else -1  # Récompense pour atteindre l'objectif ou pénalité minimale
            else:
                next_state = state  # Rester dans l'état actuel si le mouvement est invalide
                reward = -10  # Pénalité pour un déplacement invalide

            # Mise à jour de la table Q
            old_value = Q_table[state][action_index]
            next_max = np.max(Q_table[next_state])  # Meilleure valeur d'action pour l'état suivant
            Q_table[state][action_index] = old_value + learning_rate * (
                reward + discount_factor * next_max - old_value
            )

            # Passer à l'état suivant
            state = next_state

        # Réduire le taux d'exploration de façon exponentielle
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Classe pour l'interface graphique
class MazeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Q-Learning: Labyrinthe")
        self.canvas = tk.Canvas(self.master, width=500, height=500)
        self.canvas.pack()

        # Dimensions des cellules
        self.cell_width = 500 // num_cols
        self.cell_height = 500 // num_rows

        self.draw_maze()
        self.agent = None
        self.reset_agent()

        self.path_to_follow = []  # Chemin à parcourir
        self.first_run = True  # Indicateur pour le premier lancement

        # Bouton pour démarrer l'exploration
        self.go_button = tk.Button(self.master, text="GO", command=self.start_exploration)
        self.go_button.pack(side=tk.LEFT, padx=10)

        # Label pour afficher le temps écoulé
        self.time_label = tk.Label(self.master, text="Temps écoulé: 0.0 secondes")
        self.time_label.pack(side=tk.LEFT, padx=10)

    def draw_maze(self):
        # Dessiner le labyrinthe
        for row in range(num_rows):
            for col in range(num_cols):
                x1 = col * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                color = "black" if maze[row, col] == 1 else "white"  # Noir pour les murs, blanc pour les espaces libres
                if (row, col) == goal:
                    color = "green"  # Objectif
                if (row, col) == start:
                    color = "red"  # Départ
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def reset_agent(self):
        # Réinitialiser l'agent à la position de départ
        self.agent = start
        self.agent_rect = self.canvas.create_oval(
            5, 5, self.cell_width - 5, self.cell_height - 5, fill="blue"
        )

    def move_agent(self):
        # Déplacer l'agent sur le canevas
        if not self.path_to_follow:
            return

        current_state = self.path_to_follow.pop(0)

        if current_state == goal:
            # Afficher le temps écoulé une fois l'objectif atteint
            self.time_label.config(
                text=f"Temps écoulé: {time.time() - self.start_time:.2f} secondes"
            )
            return

        # Calculer le déplacement de l'agent
        dx = (current_state[1] - self.agent[1]) * self.cell_width
        dy = (current_state[0] - self.agent[0]) * self.cell_height
        self.canvas.move(self.agent_rect, dx, dy)
        self.agent = current_state

        # Continuer à déplacer l'agent après un court délai
        self.master.after(200, self.move_agent)

    def start_exploration(self):
        # Démarrer le processus d'exploration ou d'exploitation
        self.start_time = time.time()
        if self.first_run:
            train_qlearning()  # Entraîner l'algorithme
            self.path_to_follow = self.find_exploration_path(start)  # Chemin exploratoire
            self.first_run = False
        else:
            self.path_to_follow = self.find_path(start, goal)  # Chemin appris
        self.move_agent()

    def find_exploration_path(self, start):
        # Simuler une exploration aléatoire du labyrinthe
        path = []
        visited = set()
        state = start
        while state != goal:
            visited.add(state)
            path.append(state)
            valid_moves = [
                (state[0] + action[0], state[1] + action[1])
                for action in actions
                if is_valid_move(state[0] + action[0], state[1] + action[1])
            ]
            unvisited_moves = [move for move in valid_moves if move not in visited]
            state = random.choice(unvisited_moves if unvisited_moves else valid_moves)
        path.append(goal)
        return path

    def find_path(self, start, goal):
        # Trouver le chemin optimal en utilisant la table Q
        path = []
        state = start
        while state != goal:
            action_index = np.argmax(Q_table[state])  # Meilleure action selon Q-table
            action = actions[action_index]
            state = (state[0] + action[0], state[1] + action[1])
            path.append(state)
        path.append(goal)
        return path

# Lancer l'interface graphique
root = tk.Tk()
app = MazeApp(root)
root.mainloop()