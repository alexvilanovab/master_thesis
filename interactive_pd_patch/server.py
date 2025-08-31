import os
import sys
import threading
import json
import pandas as pd
import torch
import torch.nn as nn
from pythonosc import dispatcher
from pythonosc import osc_server, udp_client
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# general settings
d = ['onset_count', 'start', 'center', 'syncopation', 'balance']
n_descriptors = len(d)

# smoothness experiment
smoothness_csv_file = "../smoothness_experiment/movements.csv"
df_results = None  # Will be loaded on demand

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
model.load_state_dict(torch.load("../d2p_model/models/d2p_2.pth", weights_only=True))
model.eval()

def load_csv():
    global df_results
    if df_results is None:
        df_results = pd.read_csv(smoothness_csv_file)
        df_results["trajectory"] = df_results["trajectory"].apply(json.loads)  # Convert JSON back to list

def interpolate_handler(address, ii, istep):
    load_csv()  # Ensure CSV is loaded

    ii = int(ii)
    istep = int(istep)
    print(f"Received request for interpolation {ii}, step {istep} from {address}")

    if not (0 <= ii < len(df_results)):
        print(f"Invalid interpolation index: {ii}")
        return

    interp = df_results.iloc[ii]
    trajectory = interp["trajectory"]

    if not (0 <= istep < len(trajectory)):
        print(f"Invalid step index: {istep}")
        return

    selected_step = [round(value * 127, 2) for value in trajectory[istep]]  # Extract the 5D descriptor values and convert to MIDI with two decimals
    distance_category = interp["distance_category"]

    print(f"Sending interpolation step: {selected_step} with distance category: {distance_category}")
    client.send_message("/getinterpolation", selected_step + [distance_category])

def genpattern_handler(address, *args):
    descriptors = [arg / 127.0 for arg in args][:n_descriptors]
    descriptors = [round(x, 2) for x in descriptors]
    if descriptors[2] < descriptors[1]: descriptors[2] = descriptors[1]
    print(f"Received descriptors: {descriptors} from {address}")

    model_input = torch.tensor(descriptors, dtype=torch.float32).unsqueeze(0).to("cpu")
    with torch.no_grad():
        prediction = model(model_input)
        prediction = torch.sigmoid(prediction)
        result = prediction[0].tolist()
        result = [round(x, 2) for x in result]

    print(f"Sending response: {result}")
    client.send_message("/getpattern", result)

def start_server():
    global client, server
    disp = dispatcher.Dispatcher()
    disp.map("/genpattern", genpattern_handler)
    disp.map("/interp", interpolate_handler)

    ip = "127.0.0.1"
    receive_port = 12345
    send_port = 12346

    client = udp_client.SimpleUDPClient(ip, send_port)
    server = osc_server.ThreadingOSCUDPServer((ip, receive_port), disp)

    print(f"Serving on {ip}:{receive_port}")
    server.serve_forever()

class ReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == __file__:
            print(f"{__file__} modified. Reloading...")
            restart_program()

def restart_program():
    print("Restarting server...")
    os.execv(sys.executable, ["python"] + sys.argv)

def start_watchdog():
    event_handler = ReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

if __name__ == "__main__":
    watchdog_thread = threading.Thread(target=start_watchdog)
    watchdog_thread.daemon = True
    watchdog_thread.start()
    start_server()
