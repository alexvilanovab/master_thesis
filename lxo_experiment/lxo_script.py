import numpy as np
import math
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
# ------------------------------------------------------------------------------------------------
def v_onset_count(v_list):
    if v_list == [] * len(v_list):
        return 0
    return (np.array(v_list) > 0).sum() / len(v_list)
# ------------------------------------------------------------------------------------------------
def v_start(v_list):
    if v_onset_count(v_list) == 0:
        return 0
    s = 0
    while v_list[s] == 0:
        s += 1
    return s / (len(v_list))
# ------------------------------------------------------------------------------------------------
def v_center(v_list):
    if v_onset_count(v_list) == 0:
        return 0
    return np.mean([i + 1 for i, x in enumerate(v_list) if x > 0]) / len(v_list)
# ------------------------------------------------------------------------------------------------
def v_syncopation(v_list):
    if v_onset_count(v_list) == 0:
        return 0
    v_list = np.array(v_list)
    mw = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]  # metrical weights
    n = len(v_list)
    sync_list = np.zeros(n)
    for i, x in enumerate(v_list):
        nexti = (i + 1) % n
        vel_diff = x - v_list[nexti]
        if vel_diff > 0:  # signifficant note (1 -> 0)
            mw_diff = mw[nexti] - mw[i]
            sync_list[i] = vel_diff * mw_diff
    return (sum(sync_list) + 15) / 30
# ------------------------------------------------------------------------------------------------
def v_syncopation_awareness(v_list):
    if v_onset_count(v_list) == 0:
        return 0
    v_list = np.array(v_list)
    # iterate over pattern and find if next velocity is smaller.
    # that note should give a value as it is either a pulse reinforcement
    # or a syncopation. the degree is given by the diference in velocity
    mw = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]  # metrical weights
    n = len(v_list)
    sync_list = np.zeros(n)
    for i, x in enumerate(v_list):
        nexti = (i + 1) % n
        vel_diff = x - v_list[nexti]
        if vel_diff > 0:  # signifficant note (1 -> 0)
            mw_diff = mw[nexti] - mw[i]
            sync_list[i] = vel_diff * mw_diff
    # take into account the listener awareness based on the part being played
    sync_list[:4] = sync_list[:4] * 8
    sync_list[4:8] = sync_list[4:8] * 1
    sync_list[8:12] = sync_list[8:12] * 4
    sync_list[12:16] = sync_list[12:16] * 2
    return ((sum(sync_list) + 65)) / 115
# ------------------------------------------------------------------------------------------------
def v_evenness(v_list):
    # how well distributed are the D onsets of a pattern
    # if they are compared to a perfect D sided polygon
    # input patterns are phase-corrected to start always at step 0
    # i.e. if we have 4 onsets in a 16 step pattern, what is the distance of onsets
    # o1, o2, o3, o4 to positions 0 4 8 and 12
    # here we will use a simple algorithm that does not involve DFT computation
    # evenness is well described in [Milne and Dean, 2016] but this implementation is much simpler
    d = (np.array(v_list) > 0).sum()  # count onsets
    if d <= 1:
        return 0
    iso_angle_16 = 2 * math.pi / 16  # angle of 1/16th
    first_onset_step = [i for i, x in enumerate(v_list) if x != 0][0]
    first_onset_angle = first_onset_step * iso_angle_16
    iso_angle = 2 * math.pi / d
    # ideal positions in radians
    iso_pattern_radians = [x * iso_angle for x in range(d)]
    # real positions in radians
    pattern_radians = [i * iso_angle_16 for i, x in enumerate(v_list) if x != 0]
    # sum distortion between ideal and real
    cosines = [
        abs(math.cos(x - pattern_radians[i] + first_onset_angle))
        for i, x in enumerate(iso_pattern_radians)
    ]
    return sum(cosines) / d
# ------------------------------------------------------------------------------------------------
def v_balance(v_list):
    # balance is described in [Milne and Herff, 2020] as:
    # "a quantification of the proximity of that rhythm's
    # “centre of mass” (the mean position of the points)
    # to the centre of the unit circle."
    d = (np.array(v_list) > 0).sum()  # count onsets
    if d <= 1:
        return 0
    center = np.array([0, 0])
    iso_angle_16 = 2 * math.pi / 16
    X = [math.cos(i * iso_angle_16) for i, x in enumerate(v_list) if x != 0]
    Y = [math.sin(i * iso_angle_16) for i, x in enumerate(v_list) if x != 0]
    matrix = np.array([X, Y])
    matrix_sum = matrix.sum(axis=1)
    magnitude = np.linalg.norm(matrix_sum - center) / d
    return 1 - magnitude
# ------------------------------------------------------------------------------------------------
def v_syness(v_list):
    # compare the syncopation and the number of onsets
    # syness is higher when syncopation is high and density is small
    # NOTE: no need to divide by v_density as syncopation has
    # already taken velocity into account
    onset_count = (np.array(v_list) > 0).sum()
    if onset_count == 0:
        return 0
    res = (v_syncopation_awareness(v_list) / onset_count) / 0.6333333333333333
    # in very few cases the result can be slightly above 1
    if res > 1:
        return 1
    return res  # normalized for max syness
# ------------------------------------------------------------------------------------------------
descriptor_functions = {
    "onset_count": v_onset_count,
    "start": v_start,
    "center": v_center,
    "syncopation": v_syncopation,
    "syncopation_awareness": v_syncopation_awareness,
    "evenness": v_evenness,
    "balance": v_balance,
    "syness": v_syness,
}
# ------------------------------------------------------------------------------------------------
def describe(v_list, descriptors_to_use):
    computed_descriptors = {}
    descriptors = [d for d in descriptors_to_use if '/' not in d]
    for d in descriptors:
        computed_descriptors[d] = descriptor_functions[d](v_list)
    final_descriptors = [computed_descriptors[d] for d in descriptors]
    descriptor_combinations = [d for d in descriptors_to_use if '/' in d]
    for combination in descriptor_combinations:
        d1, d2 = combination.split('/')
        if d1 in computed_descriptors and d2 in computed_descriptors:
            val1 = computed_descriptors[d1]
            val2 = computed_descriptors[d2]
            combined_value = val1 / val2 if val2 != 0 else 0
            combined_value = min(max(combined_value, 0), 1)
            final_descriptors.append(combined_value)
        else:
            final_descriptors.append(0)
    scaled_descriptors = [int(d * 127) for d in final_descriptors]
    return tuple(scaled_descriptors)
# ------------------------------------------------------------------------------------------------
def binary_combinations(steps):
    # create all binary combinations
    # given the number of states
    # i.e. 2 = [(0,0), (1,0), (0,1), (1,1)]
    combos = []
    for event in range(2**steps):
        ensemble = []
        for i, _ in enumerate(range(steps)):
            t = ((event * 2) // (2 ** (i + 1))) % 2
            ensemble.append(t)
        combos.append(tuple(ensemble))
    return combos
# ------------------------------------------------------------------------------------------------
def generate_dataset(selected_descriptors, log=True):
    if log: print(f"Selected descriptors: {selected_descriptors}")
    # ------------------------------------------------------------------------------------------------
    n_descriptors = len(selected_descriptors)
    all_patterns = binary_combinations(16)
    all_descriptors = np.zeros([len(all_patterns), n_descriptors])
    for i, p in enumerate(all_patterns):
        all_descriptors[i] = describe(p, selected_descriptors)
    df = pd.DataFrame(all_descriptors)
    df.columns = selected_descriptors
    # save the df to a .csv file
    df.to_csv("descriptor_analysis.csv", index=True)
    # ------------------------------------------------------------------------------------------------
    unique, inverse, counts = np.unique(all_descriptors, axis=0, return_inverse=True, return_counts=True)
    if log: print(f"{len(all_descriptors) - len(unique)} are repeated")
    if log: print(f"{len(unique) / len(all_descriptors)}% are uniquely identified")
    # unique: redcued list
    # inverse: reconstruction using unique as index (len(inverse) == len(all_descriptors))
    # counts: counts of the unique
    # print patterns that have the same descriptor values thus have the same inverse value.
    # indexes of patterns that have a similar descirptor set as other(s)
    repeated = [i for i, x in enumerate(counts) if x > 1]
    # find the indexes of these indexes in the inverse list
    for x in repeated:
        rep_index = np.where(inverse == x)[0]
        for xx in rep_index:
            pass
    return n_descriptors, all_descriptors, all_patterns
# ------------------------------------------------------------------------------------------------
def train_model(model_name, n_descriptors, all_descriptors, all_patterns, log=True):
    X = torch.tensor(all_descriptors, dtype=torch.float32).to("cpu")
    y = torch.tensor(all_patterns, dtype=torch.float32).to("cpu")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    class Multiclass(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(n_descriptors, 16)
            self.act = nn.ReLU()
            self.hidden1 = nn.Linear(16, 32)
            self.act1 = nn.ReLU()
            self.hidden2 = nn.Linear(32, 64)
            self.act2 = nn.ReLU()
            self.hidden3 = nn.Linear(64, 32)
            self.act3 = nn.ReLU()
            self.output = nn.Linear(32, 16)

        def forward(self, x):
            x = self.act(self.hidden(x))
            x = self.act1(self.hidden1(x))
            x = self.act2(self.hidden2(x))
            x = self.act3(self.hidden3(x))
            x = self.output(x)
            return x

    model = Multiclass()
    model.to("cpu")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 200
    batch_size = 32
    batches_per_epoch = len(X_train) // batch_size

    best_acc = -np.inf
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    for epoch in range(n_epochs):
        model.train()

        epoch_loss = []
        epoch_acc = []

        if log:
            with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    start = i * batch_size
                    X_batch = X_train[start : start + batch_size]
                    y_batch = y_train[start : start + batch_size]

                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()

                    epoch_loss.append(float(loss))
                    epoch_acc.append(float(acc))

                    bar.set_postfix(loss=float(loss), acc=float(acc))
        else:
            for i in range(batches_per_epoch):
                start = i * batch_size
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()

                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))

        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        y_pred = torch.sigmoid(y_pred)
        predicted_pattern = torch.where(y_pred > 0.5, 1, 0)  # threshold > 0.5

        acc = torch.sum(predicted_pattern == y_test) / (predicted_pattern.size(dim=0) * predicted_pattern.size(dim=1))
        acc = float(acc)
        ce = float(ce)

        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            if log: (f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    model.load_state_dict(best_weights)

    torch.save(best_weights, "./dimred/" + model_name + ".pth")

    return best_acc, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist

    # plt.plot(train_loss_hist, label="train")
    # plt.plot(test_loss_hist, label="test")
    # plt.xlabel("epochs")
    # plt.ylabel("cross entropy")
    # plt.legend()
    # plt.show()

    # plt.plot(train_acc_hist, label="train")
    # plt.plot(test_acc_hist, label="test")
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.show()

all_descriptors = [
    "onset_count",
    "start",
    "center",
    "syncopation",
    "syncopation_awareness",
    "evenness",
    "balance",
    "syness",
]

leave_3_out_combinations = list(combinations(all_descriptors, len(all_descriptors) - 2))
leave_3_out_indexed = {i: descriptor_combination for i, descriptor_combination in enumerate(leave_3_out_combinations)}
for i, descriptor_combination in leave_3_out_indexed.items():
    model_id = i + 1
    print(f"Training model {model_id}/{len(leave_3_out_indexed)}")
    print(f"Selected descriptors: {descriptor_combination}")
    n_descriptors, all_descriptors, all_patterns = generate_dataset(descriptor_combination, log=False)
    best_acc, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = train_model(f"model_{model_id}", n_descriptors, all_descriptors, all_patterns, log=False)
    print(f"Best accuracy: ({best_acc*100:.2f}%)")
    print(f"Model saved at: ./dimred/model_{model_id}.pth")