dependencies = ['pytorch_lightning', 'captum']

from pathlib import Path
import torch
from chemprobe.models import ChemProbeEnsemble as _ChemProbeEnsemble

def ChemProbeEnsemble(**kwargs):
    """"
    ChemProbeEnsemble model, 5 models trained on separate data folds.
    attribute: (bool) if True returns attributions for each prediction (default: False)
    """
    model_dir = Path(torch.hub.get_dir())
    models = [model_dir.joinpath(f"chemprobe-ensemble/fold={i}.pt") for i in range(5)]
    try:
        model = _ChemProbeEnsemble(models, **kwargs)
    except:
        # raise ValueError
        print('Downloading model weights...')
        model_dir.joinpath("chemprobe-ensemble").mkdir(parents=True, exist_ok=False)
        # TODO TEST and store model online
        for i in range(5):
            torch.hub.download_url_to_file(f"url/fold={i}.pt", model_dir.joinpath(f"chemprobe-ensemble/fold={i}.pt"))
        model = _ChemProbeEnsemble(models, **kwargs)
    
    return model