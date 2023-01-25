dependencies = ['pytorch_lightning', 'captum', 'gdown']

from pathlib import Path
import torch
import gdown
from chemprobe.models import ChemProbeEnsemble as _ChemProbeEnsemble

def ChemProbeEnsemble(**kwargs):
    """"
    ChemProbeEnsemble model, 5 models trained on separate data folds.
    attribute: (bool) if True returns attributions for each prediction (default: False)
    """
    model_dir = Path(torch.hub.get_dir()).joinpath("chemprobe")
    models = [str(model_dir.joinpath(f"fold={i}.pt")) for i in range(5)]
    try:
        model = _ChemProbeEnsemble(models, **kwargs)
    except:
        # raise ValueError
        print('Downloading model weights...')
        model_dir.mkdir(parents=True, exist_ok=False)
        # TODO TEST and store model online
        url = "https://drive.google.com/drive/folders/1bSVeMoFu-9h7gj8ISD9yt8kC_ks9_-Y-?usp=share_link"
        gdown.download_folder(url, output=model_dir, quiet=True, use_cookies=False)
        model = _ChemProbeEnsemble(models, **kwargs)
    
    return model