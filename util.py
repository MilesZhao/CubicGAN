import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from sklearn.preprocessing import MinMaxScaler

def atom_embedding(d_elements):
    features = np.zeros((len(d_elements), 23))
    for k in d_elements:
        i = d_elements[k]
        e = Element(k)
        features[i][0] = e.Z
        features[i][1] = e.X
        features[i][2] = e.row
        features[i][3] = e.group
        features[i][4] = e.atomic_mass
        features[i][5] = float(e.atomic_radius)
        features[i][6] = e.mendeleev_no
        # features[i][7] = sum(e.atomic_orbitals.values())
        features[i][7] = float(e.average_ionic_radius)
        features[i][8] = float(e.average_cationic_radius)
        features[i][9] = float(e.average_anionic_radius)
        features[i][10] = sum(e.ionic_radii.values())
        features[i][11] = e.max_oxidation_state
        features[i][12] = e.min_oxidation_state
        features[i][13] = sum(e.oxidation_states)/len(e.oxidation_states)
        features[i][14] = sum(e.common_oxidation_states)/len(e.common_oxidation_states)
        features[i][15] = float(e.is_noble_gas)
        features[i][16] = float(e.is_transition_metal)
        features[i][17] = float(e.is_post_transition_metal)
        features[i][18] = float(e.is_metalloid)
        features[i][19] = float(e.is_alkali)
        features[i][20] = float(e.is_alkaline)
        features[i][21] = float(e.is_halogen)
        features[i][22] = float(e.molar_volume)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

def load_cubic():
    #need to change the ratio according to different dataset
    cubics_ratio = {'Fm-3m': 186344, 'F-43m': 184162,'Pm-3m':5243}
    sp2id = {'Fm-3m':0,'F-43m':1,'Pm-3m':2}
   
    prob_sp = {k:cubics_ratio[k]/sum(cubics_ratio.values()) for k in cubics_ratio}
    prob = np.zeros(len(prob_sp))
    for k in prob_sp:
        prob[sp2id[k]] = prob_sp[k]

    with open('data/ternary-dataset-pool.pkl','rb') as f:
        d = pickle.load(f)
    df = pd.read_csv('data/ternary-lable-records.csv')
    values = df.values
    ids,formulas = [],[]
    for row in values:
        ix,comp,_,symbol = row
        if symbol in cubics_ratio:
            ids.append(ix)
            formulas.append(comp)
    ids = np.array(ids).astype(str)
    np.random.shuffle(ids)
    elements = []
    for f in formulas:
        elements += list(Composition(f).as_dict().keys())
    elements = list(set(elements))
    elements.sort()
    d_elements = {}
    for i,e in enumerate(elements):
        d_elements[e]=i

    with open('data/cubic-elements-dict.json', 'w') as f:
        json.dump(d_elements, f, indent=2)
    embedding = atom_embedding(d_elements)
    np.save('data/cubic-elements-features',embedding)

    arr_sp = []
    arr_element = []
    arr_coords = []
    arr_lengths = []
    arr_angles = []
    for idx in ids:
        _,_,sp,e,coords,_,abc,angles=d[idx]
        tmp = np.rint(np.array(coords)/0.125)
        h = np.rint(np.array(angles)/30.0)
        if not np.any(np.isin(tmp, [1.0, 3.0, 5.0, 7.0])):
            arr_sp.append(sp2id[sp])
            arr_element.append([d_elements[key] for key in e])
            arr_coords.append(coords)
            arr_lengths.append(abc[0])
            arr_angles.append(angles)


    arr_sp = np.array(arr_sp).astype(int)
    arr_element = np.stack(arr_element, axis=0).astype(int)

    arr_coords = np.stack(arr_coords, axis=0).astype(float)
    m_coord_scales = np.amax(arr_coords, axis=0)/2.0

    arr_lengths = np.stack(arr_lengths, axis=0)#.reshape(len(ids),9).astype(float)
    maximum_lengths = np.amax(arr_lengths, axis=0)/2.0
    arr_angles = np.stack(arr_angles, axis=0)
    maximum_angles = np.amax(arr_angles, axis=0)/2.0

    print('for reverse\n', m_coord_scales)
    print('for reverse', maximum_lengths)
    print('for reverse', maximum_angles)    
    arr_coords = (arr_coords-m_coord_scales)/m_coord_scales
    arr_lengths = (arr_lengths-maximum_lengths)/maximum_lengths
    arr_angles = (arr_angles-maximum_angles)/maximum_angles
    print(arr_sp.shape)
    print(arr_element.shape)
    print(arr_coords.shape)
    print(arr_lengths.shape)
    print(arr_angles.shape)

    
    return (len(d_elements),len(sp2id),maximum_lengths,\
        maximum_angles,m_coord_scales,sp2id,prob),\
            (arr_sp,arr_element,arr_coords,arr_lengths,arr_angles)
    
if __name__ == '__main__':
    load_cubic()



