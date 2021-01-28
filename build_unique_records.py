import os,json
import pandas as pd
from pymatgen.core.structure import Structure
import multiprocessing as mp

import tensorflow as tf
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_samples', 5000, "# samples to generate")

def function(key):
    tree = files[key]
    for names in tree:
        name = names[0][0]

        sp = int(name.split('-')[2])
        crystal = Structure.from_file(root+'tmp-%s-cifs/'%typ+name)
        l = crystal.lattice.lengths[0]
        reduced_formula = crystal.composition.reduced_formula
        full_formula = crystal.formula.replace(' ', '')
        ctype = crystal.composition.anonymized_formula
        f = open(root + 'unique-symmetrized-cifs.csv', 'a')
        f.write('%s,%s,%s,%s,%d,%f,%d\n'%(name,\
         reduced_formula, full_formula, ctype, sp, l, len(crystal)))
        f.close()

if __name__ == '__main__':
    root = 'generated_mat/sample-%d/'%FLAGS.n_samples

    if os.path.exists(root + 'unique-symmetrized-cifs.csv'):
        os.system('rm -rf %s'%(root+'unique-symmetrized-cifs.csv'))


    f = open(root + 'unique-symmetrized-cifs.csv', 'a')
    f.write('id,reduced_formula,full_formula,type,spacegroup,gen_box,atom_num\n')
    f.close()

    typ = 'symmetrized'
    with open(root+'%s-repeating-cif-stru.json'%typ,'r') as f:
        files = json.load(f)
        keys = list(files.keys())
    print(len(keys)) 
     
    pool = mp.Pool(processes=14)
    pool.map(function, keys)


