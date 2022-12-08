dependencies = ['torch']

from pathlib import Path
import torch

def ChemProbeEnsemble():
    """"
    ChemProbeEnsemble model, 5 models trained on separate data folds.
    """
    try:
        model = torch.load("data/weights/chemprobe-ensemble.pt")
    except:
        print('Downloading model weights...')
        model_dir = Path('data/weights/')
        model_dir.mkdir(parents=True, exist_ok=False)
        torch.hub.download_url_to_file("https://drive.google.com/drive/folders/1b6qaDldujJFrW_jkgs-IRsNKyvkBVzLa?usp=sharing", "data/weights/chemprob-ensemble.pt")
        model = torch.load("data/weights/chemprobe-ensemble.pt")
    
    return model