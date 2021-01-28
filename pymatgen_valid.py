import os
import random
import warnings
warnings.filterwarnings("ignore")
from pymatgen import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
import multiprocessing as mp

import tensorflow as tf
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_samples', 5000, "# samples to generate")

if os.path.exists('generated_mat/sample-%d/tmp-symmetrized-cifs'%int(FLAGS.n_samples)):
    os.system('rm -rf generated_mat/sample-%d/tmp-symmetrized-cifs'%int(FLAGS.n_samples))
os.system('mkdir generated_mat/sample-%d/tmp-symmetrized-cifs'%int(FLAGS.n_samples))

if os.path.exists('generated_mat/sample-%d/tmp-charge-cifs'%int(FLAGS.n_samples)):
    os.system('rm -rf generated_mat/sample-%d/tmp-charge-cifs'%int(FLAGS.n_samples))
os.system('mkdir generated_mat/sample-%d/tmp-charge-cifs'%int(FLAGS.n_samples))

gen_cifs = os.listdir('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples))


def charge_check(crystal):
    # oxidation_states
    elements = list(crystal.composition.as_dict().keys())

    oxi = {}
    for e in elements:
        oxi[e] = Element(e).oxidation_states
    res = []
    if len(oxi) == 3:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k]}
                    crystal.add_oxidation_state_by_element(d)
                    if crystal.charge==0.0:
                        crystal.remove_oxidation_states()
                        res.append(d)
                    crystal.remove_oxidation_states()

    return res

def process(cif):
    sp = cif.split('---')[0].replace('#','/')
    i = int(cif.split('---')[1].replace('.cif',''))
    try:
        crystal = Structure.from_file('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples)+cif)
        
        formula = crystal.composition.reduced_formula
        sg_info = crystal.get_space_group_info(symprec=0.1)

        if sp == sg_info[0]:
            #only valid cif
            crystal.to(fmt='cif',\
             filename='generated_mat/sample-%d/tmp-symmetrized-cifs/%d-%s-%d-%d.cif'%\
             (int(FLAGS.n_samples),len(crystal),formula,sg_info[1],i),symprec=0.1)
            #charge
            res = charge_check(crystal)
            if len(res) > 0:
                crystal.to(fmt='cif',\
                  filename='generated_mat/sample-%d/tmp-charge-cifs/%d-%s-%d-%d.cif'%\
                  (int(FLAGS.n_samples),len(crystal),formula,sg_info[1],i),symprec=0.1)


    except Exception as e:
        pass


pool = mp.Pool(processes=18)
pool.map(process, gen_cifs)
pool.close()