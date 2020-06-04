# PaGraph

Scaling GNN Training on Large Graphs via Computation-aware Caching and Partitioning. Build based on DGL with PyTorch backend.

## Prerequisite

* Python 3

* PyTorch (v >= 1.3)

* DGL (v == 0.4.1)

## Prepare Dataset

* Dataset Format:

  * `adj.npz`: graph adjacancy matrix with `(vnum, vnum)` shape. Saved in `scipy.sparse` coo matrix format.

  * `labels.npy`: vertex label with `(vnum,)`. Saved in `numpy.array` format.

  * `test.npy`, `train.npy`, `val.npy`: boolean array with `(vnum, )` shape. Saved in `numpy.array`. Each element indicates whether the vertex is a train/test/val vertex.

  * (Optional) `feat.npy`: feature of vertex with `(vnum, feat-size)` shape. Saved in `numpy.array`. If not provided, will be randomly initialized (feat size is defaultly set to 600, can be changed in `Pagraph/data/get_data` line 27). 

* Convert dataset from DGL:

  ```bash
  $ python PaGraph/data/dgl2pagraph.py --dataset reddit --self-loop --out-dir /folders/to/save
  ```

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

  * For hash partition:
    
    ```bash
    $ python PaGraph/partition/hash.py --num-hops 1   --partition 2 --dataset xxx/datasetfolder
    ```

  * For dg-based partition:

    ```bash
    $ python PaGraph/partition/dg.py --num-hops 1
    --partition 2 --dataset xxx/datasetfolder
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
  $ python server/pa_server.py --dataset xxx/datasetfolder --num-workers [gpu-num] [--preprocess] [--sample]
  ```

  Note `--sample` is for enabling remote sampling.

* DGL+Cache Store Server:

  ```bash
  $ python server/cache_server.py --dataset xxx/datasetfolder --num-workers [gpu-num] [--preprocess] [--sample]
  ```

For more instructions, checkout server launch files.


### Run Trainer

* Graph Convolutional Network (GCN)

  * DGL benchmark

    ```bash
    $ python prof/profile/dgl_gcn.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] [--preprocess] [--remote-sample]
    ```

  * PaGraph

    ```bash
    $ python prof/profile/pa_gcn.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] [--preprocess] [--remote-sample]
    ```

Note: `--remote-sample` is for enabling isolation. This should be cooperated with server command `--sample`.
  
Note: multi-gpus training require `OMP_NUM_THREADS` settings, or it will show low scalability.

### Reminder

Partition is aware of GNN model layers. Please guarantee the consistency of `--num-hops`, `--preprocess` when partitioning and training, respectively. Specifically, if `--preprocess` is enabled in both server and trainer, `--num-hops` should be the `Num of model-layer - 1`. Otherwise, keep `--num-hops` the same as number of GNN layers. In our settings, GCN and GraphSAGE has 2 layers.

### Profiling

* NVProf on multi-processes command line:

  ```bash
  $ nvprof --profile-all-processes --csv --log-file %pprof.csv
  ```

* Pytorch Profiler:

  Run script in `prof/` as mentioned above.


## License

This project is under MIT License. 