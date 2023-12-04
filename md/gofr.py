import numpy as np
import ase
import numpy as np
import argparse
from ase.io import read, write
import ase.geometry
from time import time
from mpi4py import MPI
import os

parser = argparse.ArgumentParser()
# parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--nbead", type=int, default=1, help="number of beads")
# parser.add_argument("--temp", type=float, help="temperature")
parser.add_argument("--rcut", type=float, default=6.0, help="cutoff radius")
parser.add_argument("--nbin", type=int, default=200, help="number of bins")
parser.add_argument("--types", type=int, nargs=2, default=[1, 1], help="types of atoms")
parser.add_argument("--nframe", type=int, default=None, help="number of frames to read")
parser.add_argument("--nbatch", type=int, default=100, help="number of frames in a batch")
parser.add_argument("--nevery", type=int, default=1, help="every this step to read a frame")
parser.add_argument("--ndiscard", type=int, default=0, help="number of frames to discard")

args = parser.parse_args()
# id = args.id
nbead = args.nbead
# temp = args.temp
rcut = args.rcut
nbin = args.nbin
types = args.types
nframe = args.nframe
nbatch = args.nbatch
nevery = args.nevery

ndiscard = args.ndiscard
directory = "."

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
nbead = size
ibead = rank

traj_file = os.path.join(directory, "results", "{:02d}.xyz".format(ibead+1))
atoms = ase.io.read(traj_file, index=":")

type_to_atomic_number = {
    1: 8, # O has atomic number 8
    2: 1  # H has atomic number 1
}

atoms0 = atoms[0]
if nframe is None:
    nframe = len(atoms)
nloop = int(nframe/nbatch)
nframe_write = nloop * nbatch
nsub=int(nframe_write/10) 
positions0 = atoms0.get_positions()
types0 = atoms0.get_atomic_numbers()

idx_A = np.where(types0==types[0])[0]
idx_B = np.where(types0==types[1])[0]

nA = idx_A.shape[0]
nB = idx_B.shape[0]

type_numbers = [type_to_atomic_number[t] for t in types0]
atoms0.set_atomic_numbers(type_numbers)
natom = types0.shape[0]
ntype = np.unique(types0).shape[0]

dr=rcut/nbin
r_array=np.linspace(0, rcut, num=nbin, endpoint=False)+0.5*dr
r_array_10 = np.array([r_array for i in range(10)]).reshape(10, nbin)
if types[0] == types[1]:
    dists_array = np.zeros([nframe_write, int(nA*(nB-1)/2)])
    coeff = 1/4/np.pi/nA/(nB-1)*2
else:
    dists_array = np.zeros([nframe_write, int(nA*nB)])
    coeff = 1/4/np.pi/nA/nB
g_r_array = np.zeros([nframe_write, nbin])
# d_Ti_Oc = np.empty([nbead, nframe_write, idx_Ti.shape[0], 3])
# d_Ti_Oc_bead = np.empty([nframe_write, idx_Ti.shape[0], 3])
# coords = np.zeros([nbead, nframe, natom, 3])
# print(atoms0.get_chemical_symbols())

# tstart = time()
# for ibead in range(nbead):
# tstart = time()
t0 = time()
for iloop in range(nloop):
    # tloopstart = time()
    t1 = time()
    batch_idx = np.arange(iloop*nbatch*nevery, (iloop+1)*nbatch*nevery, nevery)+ndiscard
    output_idx = np.arange(iloop*nbatch, (iloop+1)*nbatch)
    # print(batch_idx)
    # traj = ase.io.read(directory+"{:02d}.xyz".format(ibead+1), index=batch_idx.tolist())
    traj = ase.io.read(traj_file, index="%d:%d:%d"%(iloop*nbatch*nevery+ndiscard, (iloop+1)*nbatch*nevery+ndiscard, nevery))
    positions = np.zeros([nbatch, natom, 3])
    prds = np.zeros([nbatch, 3])
    iframe = 0
    for atoms in traj:
        positions[iframe] = atoms.get_positions()
        for dd in range(3):
            prds[iframe][dd] = atoms.get_cell()[dd][dd]
        iframe += 1
    positionsA = positions[:, idx_A]
    positionsB = positions[:, idx_B]
    dist_batch = positionsA[:, :, None] - positionsB[:, None, :]
    dist_pbc = (dist_batch / prds[:, None, None] - np.floor(dist_batch / prds[:, None, None] + 0.5)) * prds[:, None, None]
    dist_r = np.sqrt((dist_pbc**2).sum(axis=3))

    t3 = time()
    if rank==0:
        print("Loop %d: computing pbc distances costs %.4f s."%(iloop, t3-t1))
    for ibatch in range(nbatch):
        isamp = iloop*nbatch + ibatch
        if types[0] == types[1]:
            dists_array[isamp] = dist_r[ibatch][np.triu_indices(nA, 1)]
        else:
            dists_array[isamp] = dist_r[ibatch].reshape(nA*nB)
    #print(dist_r[isamp][0][1])
    #print(dists_array[isamp])
        Vol = prds[ibatch][0]*prds[ibatch][1]*prds[ibatch][2]
        hist_r = np.histogram(dists_array[isamp], bins=nbin, range=(0, rcut), density=False)
    #print(hist_r)
    #print(hist_r[0].sum())
    #print(r_array)
        g_r_array[isamp] = hist_r[0]/r_array**2/dr*Vol*coeff

    if rank==0:
      if (isamp+1)%100==0:
        t4 = time()
        print("Computing rdf for %d samples costs %.4f s."%(isamp+1, t4-t0))

g_r_bead = np.zeros([10, nbin]) 
for isub in range(10):
  g_r_bead[isub] = g_r_array[isub*nsub:(isub+1)*nsub].mean(axis=0)
#if rank==0:
#  g_r_beads=None
g_r_beads = comm.gather(g_r_bead, root=0)

if rank==0:
  g_r = (np.array(g_r_beads, dtype="float").reshape(nbead, 10, nbin)).mean(axis=0)
  #print(g_r.shape)
  np.save(os.path.join(directory, "g"+str(types[0])+str(types[1])+".npy"), np.c_[r_array_10.reshape(10*nbin), g_r.reshape(10*nbin)].reshape(10, nbin, 2))