import os
import json
import math
import numpy as np
# np.random.seed(123)
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from util import load_cubic
import warnings
warnings.filterwarnings("ignore")
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.core.lattice import Lattice
import multiprocessing as mp

LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']

short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']

def generate_latent_inputs(lat_dim,n_samples,candidate_element_comb,aux_data):
    z = np.random.normal(0,1.0,(n_samples, lat_dim))

    p = aux_data[-1]

    label_sp = np.random.choice(aux_data[1],n_samples,p=p)

    with open('data/cubic-elements-dict.json', 'r') as f:
        e_d = json.load(f)

        exclude_ids = [e_d[e] for e in short_LaAc if e in e_d]
        other_ids = []
        for k in e_d:
            if e_d[k] not in exclude_ids:
                other_ids.append(e_d[k])

    label_elements = []
    for i in range(n_samples):
        fff = np.random.choice(other_ids,3,replace=False)
        label_elements.append(fff)
    label_elements = np.array(label_elements)

    # ix = np.random.choice(len(candidate_element_comb),n_samples)
    # label_elements = candidate_element_comb[ix]

    return [label_sp,label_elements,z]


def generate_crystal_cif(generator,lat_dim,n_samples,\
    candidate_element_comb,aux_data):
    gen_inputs = generate_latent_inputs(lat_dim,n_samples,candidate_element_comb,aux_data)
    spacegroup,formulas = gen_inputs[0],gen_inputs[1]

    sp_d = aux_data[-2]
    rsp = {sp_d[k]:k for k in sp_d}
    spacegroup = [rsp[ix] for ix in spacegroup]

    with open('data/cubic-elements-dict.json', 'r') as f:
       e_d = json.load(f)
       re = {e_d[k]:k for k in e_d}
    arr_comb = []
    for i in range(n_samples):
        arr_comb.append([re[e] for e in formulas[i]])

    coords,arr_lengths = generator.predict(gen_inputs,batch_size=1024)
    coords = coords*aux_data[4]+aux_data[4]
    coords = np.rint(coords/0.25)*0.25
    # exit(arr_lengths)
    # arr_angles = arr_angles*aux_data[3]+aux_data[3]
    arr_lengths = arr_lengths*aux_data[2]+aux_data[2]

    if os.path.exists('generated_mat/sample-%d/'%int(FLAGS.n_samples)):
        os.system('rm -rf generated_mat/sample-%d/'%int(FLAGS.n_samples))
    os.system('mkdir generated_mat/sample-%d/'%int(FLAGS.n_samples))

    if os.path.exists('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples)):
        os.system('rm -rf generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples))
    os.system('mkdir generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples))
    for i in range(n_samples):
        f = open('data/cif-template.txt', 'r')
        template = f.read()
        f.close()

        lengths = arr_lengths[i][0]
        lengths = [lengths]*3

        angles = [90.0,90.0,90.0]

        template = template.replace('SYMMETRY-SG', spacegroup[i])
        template = template.replace('LAL', str(lengths[0]))
        template = template.replace('LBL', str(lengths[1]))
        template = template.replace('LCL', str(lengths[2]))
        template = template.replace('DEGREE1', str(angles[0]))
        template = template.replace('DEGREE2', str(angles[1]))
        template = template.replace('DEGREE3', str(angles[2]))
        f = open('data/symmetry-equiv/%s.txt'%spacegroup[i].replace('/','#'), 'r')
        sym_ops = f.read()
        f.close()

        template = template.replace('TRANSFORMATION\n', sym_ops)

        for j in range(3):
            row = ['',arr_comb[i][j],arr_comb[i][j]+str(j),\
                str(coords[i][j][0]),str(coords[i][j][1]),str(coords[i][j][2]),'1']
            row = '  '.join(row)+'\n'
            template+=row

        template += '\n'
        f = open('generated_mat/sample-%d/generated-cifs/%s---%d.cif'%(int(FLAGS.n_samples),spacegroup[i].replace('/','#'),i),'w')
        f.write(template)
        f.close()

if __name__ == '__main__':


    flags = tf.compat.v1.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('num_epochs', 50, "the number of epochs for training")
    flags.DEFINE_integer('batch_size', 32, "batch size")
    flags.DEFINE_integer('lat_dim', 128, "latent noise size")
    flags.DEFINE_integer('device', 0, "GPU device")
    flags.DEFINE_integer('d_repeat', 5, "GPU device")
    flags.DEFINE_integer('n_samples', 5000, "# samples to generate")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.device)

    #load dataset and auxinary info
    AUX_DATA, DATA = load_cubic()
    candidate_element_comb = DATA[1]
    g_model = load_model('models/clean-wgan-generator-%d.h5'%(FLAGS.device), compile=False)

    generate_crystal_cif(g_model,FLAGS.lat_dim,FLAGS.n_samples,candidate_element_comb,AUX_DATA)











