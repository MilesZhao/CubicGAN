import  os 

os.system('python extract_crystal.py')
print('Dataset created')

os.system('python wgan-v2.py')
print('Trainning finished')

nn = 100# # of sampling
os.system('python generate_crystal.py --n_samples=%d'%(nn))
print('Virtural cifs generated')

os.system('python pymatgen_valid.py --n_samples=%d'%(nn))
print('Cifs checking finished')

os.system('python build_repeating_cifs_stru.py --n_samples=%d'%(nn))
print('Building of repeatted crystal structures finished')

os.system('python build_unique_records.py --n_samples=%d'%(nn))
print('Building unique crystal structures finished')






