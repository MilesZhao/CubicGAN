import os,json
import numpy as np
import multiprocessing as mp
from commons import compare_2cifs
from pymatgen.io.cif import CifFile

import tensorflow as tf
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_samples', 5000, "# samples to generate")

def base(k):
    flist = d[k]
    cnt_cif = len(flist)
    tree = []
    if cnt_cif>1:
        template = [(0,flist[0])]
        c = CifFile.from_file(source+flist[0])
        block = list(c.data.values())[0]
        tree =  [
                  [
                    [flist[0],block['_cell_length_a']]
                  ]
                ]
        for i in range(1,cnt_cif):
            flag = True
            c = CifFile.from_file(source+flist[i])
            block = list(c.data.values())[0]
            for j in range(len(template)):
                if compare_2cifs(source+flist[i], source+template[j][1]):
                    ix = template[j][0]
                    tree[ix].append([flist[i],block['_cell_length_a']])
                    flag = False
                    break
            if flag:
                    template.append((len(template), flist[i]))
                    tree.append([[flist[i],block['_cell_length_a']]])
    elif cnt_cif==1:
        c = CifFile.from_file(source+flist[0])
        block = list(c.data.values())[0]
        tree =  [
                  [
                    [flist[0],block['_cell_length_a']]
                  ]
                ]
    return k,tree

if __name__ == '__main__':

    typ = 'symmetrized'
    root = 'generated_mat/sample-%d/'%FLAGS.n_samples
    source = root+'tmp-%s-cifs/'%typ
    cifs = os.listdir(source)

    d = {}
    for fname in cifs:
        prefix = '-'.join(fname.split('-')[:3])
        if prefix not in d:
            d[prefix]=[fname]
        else:
            d[prefix].append(fname)
    print(len(cifs),len(d))

    keys = list(d.keys())

    pool = mp.Pool(processes=6)
    results = [pool.apply_async(base, args=(k,)) for k in keys]
    cnt = 0
    ans = {}
    for p in results:
        if p.get() is not None:
            k,dat = p.get()
            cnt += len(dat)
            ans[k] = dat
    print(len(ans), cnt)
    with open(root+'%s-repeating-cif-stru.json'%typ,'w') as f:
        json.dump(ans,f,indent=2)
    pool.close()




















