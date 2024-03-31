# protoplanetary-uv-plane-disk-fitting

### Import my repo
First, you should import my repository :
`git clone https://github.com/mmyas/protoplanetary-uv-plane-disk-fitting.git`

### Create a conda environment from a file (easy version)

```
cd protoplanetary-uv-plane-disk-fitting
conda env create -f CondaEnv.yml
```
You'll then want to manually install Emcee and tqdm :

```
conda activate protoplanetary-uv-plane-disk-fitting
git clone https://github.com/dfm/emcee.git
cd emcee
python3 setup.py install
cd ..
rm -rf emcee
conda deactivate
conda install -n protoplanetary-uv-plane-disk-fitting tqdm
```

### Create a conda environment and install the packages (hard and often buggy version)
Follow previous step and then you can install CUCA if you feel like it, but it's not necessary.


The last package is [galario](https://mtazzari.github.io/galario/install.html)

Fast version :
```
conda install -c conda-forge galario
```
### Import the uv table

Just put it in the protoplanetary-uv-plane-disk-fitting directory, under the name `uvtable2.txt` and then you should be good to go!

## How to use it on a SLURM computing system (such as Leftraru)?

### installation

Just download my package, install EMCEE and other modules if you need them and you should be ready to go, you just have to be carefull with the script launching system.

```
git clone https://github.com/mmyas/protoplanetary-uv-plane-disk-fittingg.git
cd protoplanetary-uv-plane-disk-fitting


git clone https://github.com/dfm/emcee.git
cd emcee
python setup.py install
cd ..
rm -rf emcee

pip install tqdm numpy --user #(etc.)
```

#### Note :
You might need to do this installation the very first time you try to launch a script, just before the actual launching (so it's also submitted with sbatch)

### Launching a script

To launch a script, the classical syntax is `sbatch /path/to/script`. It's easy, but the script must contain a detailed shebang and header, as follows :

```
#!/bin/bash
#SBATCH --partition=
#SBATCH --ntasks=
#SBATCH --ntasks-per-node=
#SBATCH --mem-per-cpu=
#SBATCH --mail-user=user@mail.ext
#SBATCH --mail-type=ALL
#SBATCH --output=
#SBATCH --error=

(options)
(installations if needed)

python OptimizationGalario.py --nwalkers 560 --iterations 1000 --nthreads 20
```

You should notice that `partitions` is the type of nodes tu use (general or slims or largemem or gpus on Leftraru), `ntasks` is the total number of threads to use, and `ntasks-per-node` is the number of cpus per node.

### Launching options

#### MPI : heavy CPU multithread
If you want to use multiple nodes and MPI support, it's not yet possible on leftraru.

#### CUDA : heavy GPU multithread

```
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --ntasks=44
#SBATCH --ntasks-per-node=44
#SBATCH --mem-per-cpu=4365
#SBATCH --mail-user=user@mail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/OptiCUDA.out
#SBATCH --error=results/OptiCUDA.err
#SBATCH --job-name=galgpu
#SBATCH --gres=gpu:2


ml fosscuda/2019b Galario/1.2.1

cd ~/protoplanetary-uv-plane-disk-fitting

python OptimizationGalario.py --nwalkers 560 --iterations 1000 --suffix _CUDA --cuda --nthreads 11
```
You'll notice that I used the Multiprocessing module (`--nthreads`). Thats because MPI is bugged on Leftraru and I can't solve it.

The reason I used 11 threads is that if you use too many, you'll end up cloging the GPUs' memory. To compensate and use the 44 core of the machine I used an option allowing each thread to have 4 cores. That makes Numpy faster and gives more RAM to the task if needed.


## What does each file do?

### `FunctionsModule.py`

It is a module with many usefull functions. Each function is described in the code.

### `TiltFinderVisibilities.py`

It finds the tilt of a Quasi-Gaussian image, *i.e.* the inclination and position angle of the image, as well as the center of the disk, using an emcee optimization.

The inc and pa will then be used to compute the visibilities with galario.

To use it, call `python3 TiltFinderVisibilities.py path/to/uvtable.txt nwalkers iterations nthreads`

Some options are available, described within the code.

### `STiltFinderSeed.py`

This is a seed file for the `TiltFinderVisibilities.py` program. It is meant to host the seed for the emcee optimization as well as its boundaries.

### `OptimizationGalario.py`

This program is meant to optimize a emcee fit in the image. It is quite custom and needs to be adapted, but it should be easy to modify.

The workings of the code are described in the code itself.

### `OptimizationGalarioMPI.py`
The same, with MPI support. Always use the MPI version if you can.

### `MergeGalOpti.py`

Can be used to merge optimization files.

It can merge files named following this pattern :
filenameN.npy, ..., filenameM.npy
where N, ..., M are consecutive numbers, and filename is whatever you want, but all the same.

`python3 MergeGalOpti.py filename N M`

### `PlotEmcee.py`

Plots the emcee optimization and a cornerplot.
