# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:34:22 2020

@author: JMMEINHARDT

This script takes a set of molecules and their SMILES keys ('SMILES Input.xlsx') 
and obtains a list of numerical descriptors ('Descriptor List.xlsx') for each molecule
using SMARTS queries and rdkit modules. The two files must be in the same working
directory as the script Python file.

The results are tabulated and recorded in an output Excel workbook ('Descriptor Data.xlsx')

'Descriptor List.xlsx' can be modified to include additional SMARTS queries.

"""

# # # IMPORT PYTHON MODULES # # #

from rdkit import Chem
from rdkit.Chem import Descriptors
import math
import pandas as pd

# # # DEFINE FUNCTIONS # # #

# this function checks for the presence of a SMARTS query in a given SMILES key and returns the incidences of the query.
def fgCounter(molecule, functional_group):
    molecule_mol = Chem.MolFromSmiles(molecule)
    funcgrp = Chem.MolFromSmarts(functional_group)
    output = (len( molecule_mol.GetSubstructMatches(funcgrp)))
    return output

# this function calculates quantities that cannot be determined by SMARTS queries, namely the number of H's, molecular weight, mass percentages, C/N ratio, and oxygen balance.
def getMolWtandAtomPercentages(molecule):
    # calculate molecular formula weight from SMILES string.
    MolWt = Descriptors.MolWt(Chem.MolFromSmiles(molecule))
    # deduce hydrogen count for the molecule.
    HeavyMolWt = Descriptors.HeavyAtomMolWt(Chem.MolFromSmiles(molecule))
    numH = math.floor(MolWt-HeavyMolWt)
    numC = fgCounter(molecule, '[#6]')
    numN = fgCounter(molecule, '[#7]')
    numO = fgCounter(molecule, '[#8]')
    # calculate mass percentages of C, H, N, O in the molecule.
    percent_C = (numC*12.011)/MolWt
    percent_H = (numH*1.008)/MolWt
    percent_N = (numN*14.007)/MolWt
    percent_O = (numO*15.999)/MolWt
    # calculate N to C ratio and oxygen balance for the molecule.
    NCRatio = numN/numC
    OxBal = (-1600/MolWt)*(2*numC+(numH/2)-numO)
    # function can return values listed below.
    return (MolWt, numH, percent_C, percent_H, percent_N, percent_O, NCRatio, OxBal)

# this function runs each molecule in molecules_df through the entire list of descriptors descriptors_df.
def analyzeMolecule(molecules_df, descriptors_df):
    # define empty list to store outputs.
    allMolecules = []
    # iterate through each molecule in molecules_df dataframe.
    for molecule in range(len(molecules_df)):
        # define temp list to store descriptor values during iteration through molecules.
        temp = []
        currentMolecule = molecules_df.iloc[molecule]['SMILES']
        # calculate values for descriptors that cannot be calculated from SMARTS queries.
        (MolWt, numH, percent_C, percent_H, percent_N, percent_O, NCRatio, OxBal) = getMolWtandAtomPercentages(currentMolecule)
        # iterate through each descriptor in descriptors_df dataframe.
        for descriptor in range(len(descriptors_df)):
            currentDescriptor = descriptors_df.iloc[descriptor]['SMARTS']
            # these conditional branches deal with descriptors that cannot be calculated from SMARTS queries.
            if pd.isnull(descriptors_df.loc[descriptor]['SMARTS']):
                # deal with 'special case' descriptors: H count, %element, MolWt, NCRatio, OxBal
                if descriptors_df.loc[descriptor]['Moiety'] == 'Molecular Weight':
                    temp.append(MolWt)
                elif descriptors_df.loc[descriptor]['Moiety'] == 'H':
                    temp.append(numH)
                elif descriptors_df.loc[descriptor]['Moiety'] == '%H':
                    temp.append(percent_H)
                elif descriptors_df.loc[descriptor]['Moiety'] == '%C':
                    temp.append(percent_C)
                elif descriptors_df.loc[descriptor]['Moiety'] == '%N':
                    temp.append(percent_N)
                elif descriptors_df.loc[descriptor]['Moiety'] == '%O':
                    temp.append(percent_O)
                elif descriptors_df.loc[descriptor]['Moiety'] == 'N/C Ratio':
                    temp.append(NCRatio)
                elif descriptors_df.loc[descriptor]['Moiety'] == 'Oxygen Balance':
                    temp.append(OxBal)
            # call fgCounter to deal with 'normal' descriptors using SMARTS queries.
            else:
                smartsCount = fgCounter(currentMolecule,currentDescriptor)
                temp.append(smartsCount)
        # append values from molecule to list containing all molecules.
        allMolecules.append(temp)
    return allMolecules

# # # MAIN JOB # # #

# load 'SMILES Input.xlsx' as a pandas dataframe molecules_df.
molecules_df = pd.read_excel('Dataset (Nitro) (Bare).xlsx')
# load 'Descriptor List.xlsx' as a pandas dataframe descriptors_df.
descriptors_df = pd.read_excel('Descriptor List.xlsx')
# analyze molecules and obtain counting descriptor list of lists.
listOfMolecules = analyzeMolecule(molecules_df, descriptors_df)

# create column titles for output workbook.
columns = list((descriptors_df.T).iloc[0,:])
#concatenate column titles, molecules_df, and moleculesWithDescriptors_df.
moleculesWithDescriptors_df = pd.DataFrame(listOfMolecules, columns=columns)
finalMerge = pd.concat([molecules_df, moleculesWithDescriptors_df], axis=1, sort=False)    
# write to output Excel workbook.
finalMerge.to_excel('Descriptor Data.xlsx')




