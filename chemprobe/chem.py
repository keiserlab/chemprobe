"""
Author: Will Connell
Date Initialized: 2022-11-29
Email: connell@keiserlab.org

Chemistry helpers.
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


# Data handling
import numpy as np
import pandas as pd

# Chem
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def smiles_to_bits(smiles, nBits):
    mols = [MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nBits) for m in mols]
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    df = pd.DataFrame(np_fps).astype(np.int8)
    return df


def generate_doses(cpds, cl, min=1e-3, max=300, num=32):
    drc = []
    for line in cl:
        for c in cpds:
            cpd_df = pd.DataFrame(
                np.geomspace(min, max, num),
                index=np.repeat(line, num),
                columns=["cpd_conc_umol"],
            )
            cpd_df["cpd_name"] = c
            cpd_df["ccl_name"] = line
            cpd_df["dose"] = cpd_df["cpd_conc_umol"]
            drc.append(cpd_df)
    return pd.concat(drc, ignore_index=False)