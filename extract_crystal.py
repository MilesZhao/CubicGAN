import os
import re
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from pymatgen.io.cif import CifFile,CifParser
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']

root = 'data/trn-cifs/'
valid_files = os.listdir(root)


def function(file):
    # file = str(file)+'.cif'
    cif = CifFile.from_file(root+file)
    data = cif.data
    formula = list(data.keys())[0]
    block = list(data.values())[0]
    bases = block['_atom_site_type_symbol']
    #remove materials with La AND Ac rows
    #remove materials having more than three base atoms 
    occu = np.array(block['_atom_site_occupancy']).astype(float)

    if len(bases)==3 and len(set(bases))==3 and all(occu == 1.0):
        # print(file,bases)

        xs = np.array(block['_atom_site_fract_x']).reshape((3,1))
        ys = np.array(block['_atom_site_fract_y']).reshape((3,1))
        zs = np.array(block['_atom_site_fract_z']).reshape((3,1))
        coords = np.hstack([xs,ys,zs]).astype(float)
        lengths = np.array([block['_cell_length_a'],block['_cell_length_b'],block['_cell_length_c']]).astype(float)
        angles = np.array([block['_cell_angle_alpha'],block['_cell_angle_beta'],block['_cell_angle_gamma']]).astype(float)
        lattice = Lattice.from_parameters(
                lengths[0],
                lengths[1],
                lengths[2],
                angles[0],
                angles[1],
                angles[2],
            )
        matrix = lattice.matrix

        a = [
            formula,
            block['_symmetry_Int_Tables_number'],
            block['_symmetry_space_group_name_H-M'],
            bases,
            coords,
            matrix,
            lengths,
            angles
        ]

        b = [
            file.replace('.cif',''),
            formula,
            block['_symmetry_Int_Tables_number'],
            block['_symmetry_space_group_name_H-M'],
            ]
        return a,b


pool = mp.Pool(processes=24)
results = [pool.apply_async(function, args=(file,)) for file in valid_files]

d = {}
ternary_uniq_sites = []

for p in results:
    if p.get() is not None:
        a,b = p.get()
        d[b[0]] = a
        ternary_uniq_sites.append(b)

print(len(ternary_uniq_sites))

with open('data/ternary-dataset-pool.pkl','wb') as f:
    pickle.dump(d, f, protocol=4)


df = pd.DataFrame(np.array(ternary_uniq_sites), columns=['id','formula','spid','spsym'])
df.to_csv('data/ternary-lable-records.csv', index=False)

















