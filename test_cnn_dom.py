"""Test the threat model to the cnn-dom detector. 

Run this script:
nohup python -u test_cnn_dom.py > test_cnn_dom.out 2>&1 & disown
"""

import html
import json
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from src.optimizer import Optimizer

CWD = os.getcwd()
PHISH_DETECTORS_BASE_DIR = os.path.join(os.path.dirname(CWD), "phish_detectors")
sys.path.insert(0, os.path.join(PHISH_DETECTORS_BASE_DIR, "src"))

from phish_detectors.data.preprocess import get_dom
from phish_detectors.models import CNNClassifier
from phish_detectors.utils.interface import PhishDetector

##################################################
########## Parameters (can change)      ##########
##################################################
# Options
MODEL = "cnn"
REPRE = "dom"
SEQUENCE_LENGTH = 1000  # train_cnn_dom.out
VOCABULARY_SIZE = 178  # train_cnn_dom.out
NUM_ROUNDS = 10
# Directories
DATA_DIR = "/media/volume/sdb/detector_data/"
DETECTOR_MODEL_DIR = os.path.join(PHISH_DETECTORS_BASE_DIR, f"outputs/{MODEL}_{REPRE}")
DETECTOR_MODEL_PATH = os.path.join(DETECTOR_MODEL_DIR, "model6.pth")
OUTPUT_DIR = f"outputs/attack_{MODEL}_{REPRE}"
ADV_SAMPLE_DIR = os.path.join(OUTPUT_DIR, "adv")
##################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADV_SAMPLE_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)

# Get cpu or gpu device for training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


def preproc(html_str):
    updated = False
    # preproc_xhtml if needed
    if re.search("xhtml", html_str, re.IGNORECASE):
        html_str = re.sub(r"<(\w)+:", "<", html_str)
        html_str = re.sub(r"</(\w)+:", "</", html_str)
        updated = True

    if len(re.findall(r"&(#?)(\d{1,5}|\w{1,8});", html_str)) > 0:
        html_str = html.unescape(html_str)
        updated = True

    return html_str, updated


class DetectorWrapper:
    """A wrapper that convert our detector to the form that this repo requires."""

    def __init__(self, detector: PhishDetector):
        self.detector = detector

    def classify(self, html_url: tuple[BeautifulSoup, str]) -> float:
        return self.detector.score(str(html_url[0]))


# Load Model
pt_model = CNNClassifier(
    sequence_length=SEQUENCE_LENGTH,
    vocabulary_size=VOCABULARY_SIZE,
    emb_dim=128,
    dropout_rate=0.5,
    filters=256,
    kernel_size=5,
    pool_size=5,
    strides=4,
    neurons1=128,
    neurons2=64,
)
pt_model.load_state_dict(torch.load(DETECTOR_MODEL_PATH))
pt_model = pt_model.to(DEVICE)
print(pt_model)
pt_model.eval()
detector = PhishDetector(
    preprocess=get_dom,
    transform=torch.load(os.path.join(DETECTOR_MODEL_DIR, "transform.pth")),
    model=pt_model,
)
model = DetectorWrapper(detector)

# Prepare data (detected phish websites)
test_idx = np.load(os.path.join(DETECTOR_MODEL_DIR, "test_TP_idx.npy"))
annotations = pd.read_csv(os.path.join(DATA_DIR, "annotations.csv"))
html_paths = annotations["file"].iloc[test_idx].to_list()
file_ids = [os.path.split(os.path.dirname(html_path))[1] for html_path in html_paths]
url_paths = [
    os.path.join(os.path.dirname(html_path), "url.txt") for html_path in html_paths
]

# Attacking!
all_traces = []
for file_id, html_path, url_path in tqdm(
    zip(file_ids, html_paths, url_paths), total=len(file_ids)
):

    with open(html_path, "r", encoding="utf-8") as f:
        html_str, updated = preproc(f.read())
    with open(url_path, "r") as f:
        url = f.read()
    html_obj = BeautifulSoup(html_str, "html.parser")
    input_sample = (html_obj, url)

    try:
        optimizer = Optimizer(model, NUM_ROUNDS, save_only_best=True, target_score=0.5)
        best_score, adv_example, num_queries, run_time, scores_trace = (
            optimizer.optimize(input_sample)
        )
    except Exception as e:
        all_traces.append([])
        print(file_id, str(e))
        continue

    # record the process
    all_traces.append(scores_trace)

    # if evade, store the adv sample.
    if best_score < 0.5:
        html_adv_obj, adv_url = adv_example
        adv_folder = os.path.join(ADV_SAMPLE_DIR, file_id)
        os.makedirs(adv_folder, exist_ok=True)
        with open(os.path.join(adv_folder, "index.html"), "w", encoding="utf-8") as f:
            f.write(str(html_adv_obj.prettify()))
        with open(os.path.join(adv_folder, "url.txt"), "w", encoding="utf-8") as f:
            f.write(url)

with open(os.path.join(OUTPUT_DIR, "all_traces.json"), "w") as f:
    json.dump(all_traces, f)


# count number of evasions
with open(os.path.join(OUTPUT_DIR, "all_traces.json"), "r") as f:
    all_traces = json.load(f)
num_manipulations = np.zeros(len(all_traces))
for i, trace in enumerate(all_traces):
    if len(trace) == 0:  # error
        num_manipulations[i] = np.inf
    elif trace[-1][1] < 0.5:  # evade
        num_manipulations[i] = len(trace) - 1
    else:  # not evade
        num_manipulations[i] = np.inf
n_actions = np.arange(10 + NUM_ROUNDS + 1)
n_evades = np.asarray([(num_manipulations <= k).sum() for k in n_actions])
plt.figure()
plt.plot(n_actions, n_evades)
plt.title("evaded examples vs manipulations")
plt.xlabel("number of manipulations")
plt.ylabel("number of evaded examples")
plt.savefig(os.path.join(OUTPUT_DIR, "evade_manipulation.png"))
