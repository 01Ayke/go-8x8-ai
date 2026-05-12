# Import du fichier d'exemples


def get_raw_data_go(filename="samples-8x8.json.gz"):
    """Returns the set of samples from the local file or download it if it does not exists"""
    import gzip, os.path
    import json

    raw_samples_file = filename

    if not os.path.isfile(raw_samples_file):
        print("File", raw_samples_file, "not found, I am downloading it...", end="")
        import urllib.request

        urllib.request.urlretrieve(
            "https://www.labri.fr/perso/lsimon/static/inge2-ia/samples-8x8.json.gz",
            "samples-8x8.json.gz",
        )
        print(" Done")

    with gzip.open(filename) as fz:
        data = json.loads(fz.read().decode("utf-8"))
    return data


#
# data = get_raw_data_go()
# print("We have", len(data), "examples")
#

import torch
import numpy as np

# Cette cellule définit seulement les fonctions de prédiction.
# Elle peut être exécutée avant l'entraînement : les variables model_final,
# device, sample_to_tensor et augment_board seront utilisées plus tard,
# une fois créées dans les cellules suivantes.


def position_predict(black_stones, white_stones, list_of_moves=None, depth=None):
    if list_of_moves is None:
        list_of_moves = []
    sample = {
        "black_stones": black_stones,
        "white_stones": white_stones,
        "list_of_moves": list_of_moves,
    }
    if depth is not None:
        sample["depth"] = depth

    x = sample_to_tensor(sample)

    # Moyenne sur les 8 symétries du plateau : la prédiction est plus stable.
    boards = augment_board(x)
    X = torch.tensor(np.array(boards, dtype=np.float32), dtype=torch.float32).to(device)

    model_final.eval()
    with torch.no_grad():
        predictions = model_final(X)

    prediction = predictions.mean().item()
    prediction = max(0.0, min(1.0, prediction))
    return float(prediction)


def create_result_file(newdata, filename="my_predictions.txt"):
    resultat = []
    for d in newdata:
        p = position_predict(
            d["black_stones"],
            d["white_stones"],
            d.get("list_of_moves", []),
            d.get("depth", None),
        )
        resultat.append(p)

    with open(filename, "w") as f:
        for p in resultat:
            f.write(str(p) + "\n")

    print("Fichier", filename, "créé avec", len(resultat), "prédictions.")


import numpy as np
import torch

BOARD_SIZE = 8
NB_CHANNELS = 6

# Ordre des 6 canaux :
# 0 : pierres blanches
# 1 : pierres noires
# 2 : joueur au trait, 1.0 si noir joue, 0.0 si blanc joue
# 3 : cases vides
# 4 : cases vides voisines d'une pierre noire
# 5 : cases vides voisines d'une pierre blanche
WHITE_CHANNEL = 0
BLACK_CHANNEL = 1
TURN_CHANNEL = 2
EMPTY_CHANNEL = 3
BLACK_NEIGHBOR_CHANNEL = 4
WHITE_NEIGHBOR_CHANNEL = 5


def name_to_coord(s):
    if s == "PASS":
        return None
    indexLetters = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 8,
    }
    col = indexLetters[s[0]]
    lin = int(s[1:]) - 1
    return col, lin


def inside_board(lin, col):
    return 0 <= lin < BOARD_SIZE and 0 <= col < BOARD_SIZE


def add_empty_neighbors_channel(x, stone_channel, neighbor_channel):
    """Marque les cases vides voisines orthogonalement des pierres demandées."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for lin, col in np.argwhere(x[stone_channel] == 1.0):
        for dlin, dcol in directions:
            nlin, ncol = lin + dlin, col + dcol
            if inside_board(nlin, ncol) and x[EMPTY_CHANNEL, nlin, ncol] == 1.0:
                x[neighbor_channel, nlin, ncol] = 1.0


def sample_to_tensor(sample):
    """Encode un exemple en tenseur numpy de taille [6, 8, 8]."""
    x = np.zeros((NB_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for stone in sample["white_stones"]:
        coord = name_to_coord(stone)
        if coord is not None:
            col, lin = coord
            if inside_board(lin, col):
                x[WHITE_CHANNEL, lin, col] = 1.0

    for stone in sample["black_stones"]:
        coord = name_to_coord(stone)
        if coord is not None:
            col, lin = coord
            if inside_board(lin, col):
                x[BLACK_CHANNEL, lin, col] = 1.0

    # Longueur paire de list_of_moves => noir joue.
    # Longueur impaire => blanc joue.
    if len(sample.get("list_of_moves", [])) % 2 == 0:
        x[TURN_CHANNEL, :, :] = 1.0

    occupied = (x[WHITE_CHANNEL] == 1.0) | (x[BLACK_CHANNEL] == 1.0)
    x[EMPTY_CHANNEL] = (~occupied).astype(np.float32)

    add_empty_neighbors_channel(x, BLACK_CHANNEL, BLACK_NEIGHBOR_CHANNEL)
    add_empty_neighbors_channel(x, WHITE_CHANNEL, WHITE_NEIGHBOR_CHANNEL)

    return x


def augment_board(x):
    """4 rotations x 2 miroirs = 8 symétries géométriques."""
    augmented = []
    for k in range(4):
        rotated = np.rot90(x, k=k, axes=(1, 2)).copy()
        augmented.append(rotated)
        flipped = np.flip(rotated, axis=2).copy()
        augmented.append(flipped)
    return augmented


def color_swap_tensor(x, y):
    """
    Inversion noir/blanc correctement adaptée aux 6 canaux.
    On échange les pierres et les voisinages, mais les cases vides restent identiques.
    """
    x_swap = x.copy()
    x_swap[WHITE_CHANNEL] = x[BLACK_CHANNEL]
    x_swap[BLACK_CHANNEL] = x[WHITE_CHANNEL]
    x_swap[BLACK_NEIGHBOR_CHANNEL] = x[WHITE_NEIGHBOR_CHANNEL]
    x_swap[WHITE_NEIGHBOR_CHANNEL] = x[BLACK_NEIGHBOR_CHANNEL]
    x_swap[EMPTY_CHANNEL] = x[EMPTY_CHANNEL]

    # L'inversion couleur est utilisée seulement pour les profondeurs paires,
    # donc le joueur au trait reste cohérent.
    x_swap[TURN_CHANNEL] = x[TURN_CHANNEL]

    return x_swap, np.float32(1.0 - y)


def build_dataset(samples, augment=True, use_color_swap=True):
    X, y = [], []

    for sample in samples:
        x = sample_to_tensor(sample)
        target = np.float32(sample["black_wins"] / sample["rollouts"])

        variants = [(x, target)]

        # L'inversion noir/blanc est gardée seulement quand la profondeur est paire.
        # Cela évite de fabriquer des positions incohérentes avec l'ordre de jeu.
        depth = sample.get("depth", len(sample.get("list_of_moves", [])))
        if use_color_swap and depth % 2 == 0:
            variants.append(color_swap_tensor(x, target))

        for xb, yb in variants:
            if augment:
                for x_aug in augment_board(xb):
                    X.append(x_aug)
                    y.append(yb)
            else:
                X.append(xb)
                y.append(yb)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


# # Split train/test pour évaluer le modèle sur un jeu de test séparé.
# TEST_RATIO = 0.2
# rng = np.random.default_rng(42)
# indices = rng.permutation(len(data))
# test_size = max(1, int(len(data) * TEST_RATIO))
# if test_size >= len(data):
#     test_size = len(data) - 1
#
# test_indices = indices[:test_size]
# train_indices = indices[test_size:]
#
# train_data = [data[i] for i in train_indices]
# test_data = [data[i] for i in test_indices]
#
# print("Exemples originaux utilisés pour l'entraînement :", len(train_data))
# print("Exemples originaux utilisés pour le test :", len(test_data))
#
# X_train, y_train = build_dataset(train_data, augment=True, use_color_swap=True)
#
# print("X_train :", X_train.shape)
# print("y_train :", y_train.shape)
#
# assert X_train.shape[1:] == (NB_CHANNELS, BOARD_SIZE, BOARD_SIZE)
#
# # Sauvegarde utile si tu relances seulement la partie entraînement.
# np.save("X_train.npy", X_train)
# np.save("y_train.npy", y_train)
# print("Données préparées et sauvegardées.")
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
import os

# Reproductibilité minimale.
torch.manual_seed(42)
np.random.seed(42)


class GoValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(NB_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Si la cellule de préparation n'a pas été relancée, on recharge les fichiers sauvegardés.
if __name__ == "__main__":
    try:
        X_train
        y_train
    except NameError:
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    model = GoValueNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    MODEL_PATH = "go_cnn_6_canaux_final.pt"

    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model_final = model
        print(f"Modèle existant chargé : {MODEL_PATH}")
    else:
        TARGET_TRAIN_LOSS = 0.02
        epochs = 15
        train_losses = []

        best_loss = float("inf")
        best_state = None

        print("Début de l'entrainement")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            print(f"Epoch {epoch + 1}/{epochs} → Train Loss: {epoch_train_loss:.5f}")

            if epoch_train_loss < best_loss:
                best_loss = epoch_train_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, "go_cnn_6_canaux_best.pt")

            if epoch_train_loss <= TARGET_TRAIN_LOSS:
                print(
                    f"Objectif atteint : train loss = {epoch_train_loss:.5f} <= {TARGET_TRAIN_LOSS}"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model_final = model
        torch.save(model_final.state_dict(), MODEL_PATH)

        print("Meilleure train loss :", best_loss)
        print(f"Modèle sauvegardé : {MODEL_PATH}")

    # Prédictions sur tout le dataset de test et comparaison avec les vraies valeurs.
    test_predictions = []
    test_targets = []

    for sample in test_data:
        pred = position_predict(
            sample["black_stones"],
            sample["white_stones"],
            sample.get("list_of_moves", []),
            sample.get("depth", None),
        )
        target = sample["black_wins"] / sample["rollouts"]
        test_predictions.append(pred)
        test_targets.append(target)

    test_predictions = np.array(test_predictions, dtype=np.float32)
    test_targets = np.array(test_targets, dtype=np.float32)

    test_mae = np.mean(np.abs(test_predictions - test_targets))
    test_mse = np.mean((test_predictions - test_targets) ** 2)

    print("Évaluation sur le dataset de test")
    print("Nombre d'exemples test :", len(test_data))
    print("MAE test :", float(test_mae))
    print("MSE test :", float(test_mse))

    # Pour générer le fichier de prédictions sur les nouvelles données :
    create_result_file(get_raw_data_go("positions-to-evaluate-8x8-2026.json.gz"))
