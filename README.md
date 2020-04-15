# PaGraph

Scaling GNN Training on Large Graphs via Computation-aware Caching and Partitioning. Build based on DGL with PyTorch backend.

## Prerequisite

* Python 3

* PyTorch (v >= 1.3)

* DGL (v == 0.4.1)

## Prepare Dataset

* For randomly generated dataset:

  * Use [PaRMAT](https://github.com/farkhor/PaRMAT) to generate a graph:

    ```bash
    $ ./PaRMAT -noDuplicateEdges -undirected -threads 16 -nVertices 10 -nEdges 25 -output /path/to/datafolder/pp.txt

    ```
  
  * Generate random features, labels, train/val/test datasets:

    ```bash
    $ python data/preprocess.py --ppfile pp.txt --gen-feature --gen-label --gen-set --dataset xxx/datasetfolder
    ```

    This may take a while to generate all of these.

* Generating partitions (naive partition):

  ```bash
  $ python partition/fastbuilding_old.py --num-hops 1 --partition 2 --dataset xxx/datasetfolder

  ```

## Run

### Install

```bash
$ python setup.py develop
$ python
>> import PaGraph
>> 
``` 

### Launch Graph Server

* PaGraph Store Server:

  ```bash
  $ python server/pa_server.py --dataset xxx/datasetfolder --num-workers [gpu-num] [--preprocess] [-sample]
  ```

  Note `--sample` is for enabling remote sampling.

* DGL+Cache Store Server:

  ```bash
  $ python server/cache_server.py --datset xxx/datasetfolder --num-workers [gpu-num] [--preprocess] [--sample]
  ```

For more instructions, checkout server launch files.


### Run Trainer

* Graph Convolutional Network (GCN)

  * DGL benchmark

    ```bash
    $ python prof/profile/dgl_orig.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] [--preprocess]
    ```

  * PaGraph

    ```bash
    $ python prof/profile/pa.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] [--preprocess]
    ```

  * Isolation

    ```bash
    $ python prof/profile/dgl_iso.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] [--preprocess]
    ```

    ```bash
    $ python prof/profile/pa_iso.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] [--preprocess]
    ```
  
Note: multi-gpus training require `OMP_NUM_THREADS` settings, or it will show low scalability.

### Profiling with NVProf

* multi-processes command line:

  ```bash
  $ nvprof --profile-all-processes --csv --log-file %pprof.csv
  ```

## License

This project is under MIT License. 