# PaGraph

Graph Neural Network Framework on Large Scaled Graph Dataset with Multi-GPUs training, partitioning and caching.

## Prerequisite

* Python 3

* PyTorch

* DGL v = 0.4

* numba (conda install numba)

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
  $ python partition/partition.py --num-hop 1 --dataset xxx/datasetfolder

  ```

## Run

### Launch Graph Server

* PyTorch

  ```bash
  $ python server/pytorch/launch_server.py --num-workers 4 --preprocess --dataset xxx/datasetfolder
  ```

* MXNet

  ```bash
  DGLBACKEND=mxnet python server/mxnet/launch_server.py --dataset xxx/datasetfolder --num-workers 3
  ```

### Run Client Trainer

* Run w/o Partitioning

  * PyTorch

    ```bash
    $ DGLBACKEND=pytorch python examples/pytorch/gcn_nccl_nssc.py --gpu 0,1 --num-neighbors 2 --batch-size 30000 --dataset /path/to/datasetfolder

    $ DGLBACKEND=pytorch python examples/pytorch/eval.py --gpu 0 --arch gcn-nssc --batch-size 512 --epoch 30 --feat-siz 602 --dataset /path/to/datasetfolder
    ```
  
  * MXNet

    ```bash
    DGLBACKEND=mxnet python examples/mxnet/launch.py -n 1 -s 1 --launcher local python examples/mxnet/gcn_client_nssc.py --batch-size 2500 --test-batch-size 5000 --n-epochs 60 --graph-name reddit --num-neighbors 2 --n-hidden 128 --dropout 0.2 --weight-decay 0 --num-gpus 1
    ```

    ```bash
    DGLBACKEND=mxnet python examples/mxnet/launch.py -n 2 -s 1 --launcher local python examples/mxnet/gcn_nssc.py --dataset /home/lzq/data/graph-gen/3mv150me --ngpu 2 --batch-size 2500 --n-epochs 60 --num-neighbors 2
    ```

* Run on Reddit-small dataset:

  * Pytorch

    ```bash
    $ DGLBACKEND=pytorch python examples/pytorch/pa_gcn_nssc.py --gpu 0,1 --num-neighbors 2 --batch-size 6000 --dataset /home/lzq/data/reddit-small --feat-size 602 --n-classes 41 --preprocess
    ```

    ```bash
    $ DGLBACKEND=pytorch OMP_NUM_THREADS=2 python examples/pytorch/pa_gcn_nssc.py --num-neighbors 2 --batch-size 6000 --feat-size 600 --gpu 0,1 --dataset /home/lzq/data/livejournal --preprocess
    ```


### Profiling with NVProf

* multi-processes command line:

  ```bash
  $ nvprof --profile-all-processes --csv --log-file %pprof.csv
  ```

## License

This project is under MIT License. 