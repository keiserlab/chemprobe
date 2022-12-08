dependencies = ['torch']

from pathlib import Path
import torch

def ChemProbeEnsemble():
    """"
    ChemProbeEnsemble model, 5 models trained on separate data folds.
    """
    model_dir = Path(__file__).resolve().parent.joinpath("data/weights")
    try:
        model = torch.load(model_dir.joinpath("chemprobe-ensemble.pt"))
    except:
        print('Downloading model weights...')
        model_dir.mkdir(parents=True, exist_ok=False)
        # TODO store model somewhere else, not google drive
        torch.hub.download_url_to_file("https://drive.google.com/file/d/1DYcE46rvbcLgIUyLOI1yD8vbxTk827_8", model_dir.joinpath("chemprobe-ensemble.pt"))
        model = torch.load("data/weights/chemprobe-ensemble.pt")
    
    return model