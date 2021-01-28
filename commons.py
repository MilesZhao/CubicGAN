import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.cif import CifFile


def compare_2cifs(f1,f2):
    # try:
    c1 = CifFile.from_file(f1)
    data = c1.data
    block = list(data.values())[0]
    xs = np.array(block['_atom_site_fract_x'])
    ys = np.array(block['_atom_site_fract_y'])
    zs = np.array(block['_atom_site_fract_z'])
    coords1 = np.hstack([xs,ys,zs]).astype(float)
    site_type1 = np.array(block['_atom_site_label'])

    c2 = CifFile.from_file(f2)
    data = c2.data
    block = list(data.values())[0]
    xs = np.array(block['_atom_site_fract_x'])
    ys = np.array(block['_atom_site_fract_y'])
    zs = np.array(block['_atom_site_fract_z'])
    coords2 = np.hstack([xs,ys,zs]).astype(float)
    site_type2 = np.array(block['_atom_site_label'])

    return np.all(site_type1 == site_type2) and\
     np.array(coords1==coords2).astype(int).sum() == len(coords1)

def ionic_radii():
    elements =[
        'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',\
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',\
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',\
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La',
        'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',\
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',\
        'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm',\
        'Md','No','Lr'
    ]
    d = {}
    no_radii_ele = []
    for e in elements:
        r = Element(e).ionic_radii
        if len(r)!=0:
            d[e] = r 
        else:
            no_radii_ele.append(e)

    return d, no_radii_ele

def charge_check(crystal, ionic):
    d_radii,no_radii_ele = ionic
    common_oxidation_states = {}
    for e in d_radii:
        common_oxidation_states[e] = tuple(d_radii[e].keys())

    # oxidation_states
    elements = list(crystal.composition.as_dict().keys())
    if set(elements)&set(no_radii_ele):
        return []
    oxi = {}
    for e in elements:
        oxi[e] = common_oxidation_states[e]

    tmp_oxidates = []
    if len(oxi) == 3:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k]}
                    crystal.add_oxidation_state_by_element(d)
                    if crystal.charge==0.0:
                        crystal.remove_oxidation_states()
                        tmp_oxidates.append(d)
                        continue
                    crystal.remove_oxidation_states()
    elif len(oxi) == 4:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    for l in range(len(oxi[elements[3]])):
                        d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k], elements[3]:oxi[elements[3]][l]}
                        crystal.add_oxidation_state_by_element(d)
                        if crystal.charge==0.0:
                            crystal.remove_oxidation_states()
                            tmp_oxidates.append(d)
                            continue
                        crystal.remove_oxidation_states()
    return tmp_oxidates

def charge_pauling_check(crystal,ionic):
    
    d_radii,no_radii_ele = ionic
    common_oxidation_states = {}
    for e in d_radii:
        common_oxidation_states[e] = tuple(d_radii[e].keys())

    # oxidation_states
    elements = list(crystal.composition.as_dict().keys())
    if set(elements)&set(no_radii_ele):
        return [],[]
    oxi = {}
    for e in elements:
        oxi[e] = common_oxidation_states[e]

    tmp_oxidates = []
    if len(oxi) == 3:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k]}
                    crystal.add_oxidation_state_by_element(d)
                    if crystal.charge==0.0:
                        crystal.remove_oxidation_states()
                        tmp_oxidates.append(d)
                        continue
                    crystal.remove_oxidation_states()
    elif len(oxi) == 4:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    for l in range(len(oxi[elements[3]])):
                        d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k], elements[3]:oxi[elements[3]][l]}
                        crystal.add_oxidation_state_by_element(d)
                        if crystal.charge==0.0:
                            crystal.remove_oxidation_states()
                            tmp_oxidates.append(d)
                            continue
                        crystal.remove_oxidation_states()

    strategy = CrystalNN()
    cn2radius_ratio = {3:0.155, 4:0.225, 6:0.414, 7:0.592, 8:0.732, 9:0.732, 12:1.0}
    res = []
    # if len(tmp_oxidates)==0:
    #     return res
    for oxidates in tmp_oxidates:
        crystal.add_oxidation_state_by_element(oxidates)
        flag = True
        for i,site in enumerate(crystal):
            center_symbol = site.specie.symbol
            if oxidates[center_symbol]>0:#for now, some repeatings perhaps
                cation_radius = d_radii[center_symbol][oxidates[center_symbol]]
                siw = strategy.get_nn_info(crystal, i)
                if len(siw) in cn2radius_ratio:
                    radius_ratio = cn2radius_ratio[len(siw)]
                    for item in siw:
                        neighbor_symbol = item['site'].specie.symbol
                        anion_radius = d_radii[neighbor_symbol][oxidates[neighbor_symbol]]
                        if cation_radius/anion_radius < radius_ratio:
                            flag = False
                            break
        if flag:
            res.append(oxidates)
        crystal.remove_oxidation_states()

    return tmp_oxidates,res



