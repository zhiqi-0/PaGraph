import os
import sys
import torch
import torch.multiprocessing as multiprocessing
from . import SampleLoader
from . import _utils
import queue
import time
import threading

class OverLap(object):
    r"""
    launch a deamon processer to overlap data loading from CPU and computation on GPU
    """

    def __init__(self, sampler, num_workers=0, load_local=False):
        self._sampler = sampler
        self._num_workers = num_workers
        self._load_local = load_local

    def __iter__(self):
        if self._num_workers == 0:
            return _MainProcessLoader(self)
        else:
            return _DaemonProcessLoader(self)

class OverLapInitSamplerAtWorker(object):
    def __init__(self, graph, rank, one2all=False, load_local=False):
        self._graph = graph
        self._rank = rank
        self._one2all = one2all
        self._sampler = None
        self._num_workers = 1
        self._epoch = 0
        self._load_local = load_local

    def __iter__(self):
        self._epoch += 1
        return _DaemonProcessLoader(self)

class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        if loader._sampler is not None:
            self._sampler = loader._sampler
            self._num_workers = loader._num_workers
            self._load_local = loader._load_local
        else:
            self._graph = loader._graph
            self._rank = loader._rank
            self._one2all = loader._one2all
            self._num_workers = 1
            self._load_local = loader._load_local

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class _MainProcessLoader(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_MainProcessLoader, self).__init__(loader)
        assert self._num_workers == 0
    
    def __next__(self):
        return next(self._sampler)


class _DaemonProcessLoader(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_DaemonProcessLoader, self).__init__(loader)
        assert self._num_workers == 1

        self._mp_context = multiprocessing
        self._worker_result_queue = self._mp_context.Queue()
        self._num_elem_in_queue = 4 # how much the worker faster than main processor

        # signal to shutdown the worker
        self._worker_done_event = self._mp_context.Event()
        self._shutdown = False

        # create the index to maintain the order
        self._send_idx = 0 # idx of the next task to be sent to workers
        self._rcvd_idx = 0 # idx of the next task to be returned in __next__
        self._tasks_outstanding = 0

        # create the worker processer
        self._index_queue = self._mp_context.Queue()
        self._worker = self._mp_context.Process(
            target=_utils.worker._worker_loop_init_sampler,
            args=(loader, self._index_queue,
                  self._worker_result_queue, self._worker_done_event, 
                  self._num_workers,)
        )
        self._worker.daemon = True
        self._worker.start()
        self._worker_status = True

        # may use load local thread to decouple the data_queue and worker_result_queue
        if self._load_local:
            self._load_local_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()
            load_local_thread = threading.Thread(
                target=_utils.load_local._load_local_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._load_local_thread_done_event,)
            )
            load_local_thread.daemon = True
            load_local_thread.start()
            self._load_local_thread = load_local_thread
        else:
            self._data_queue = self._worker_result_queue

        # prime the prefetch loop
        for _ in range(self._num_elem_in_queue*self._num_workers):
            self._try_put_index()
        
    
    def __next__(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                if self._worker_status:
                    break
                self._rcvd_idx += 1
            else:
                self._shutdown_workers()
                raise StopIteration

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            
            # # Check for _IterableDatasetStopIteration, the end of iterator
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                self._shutdown_worker()
                self._try_put_index()
                continue
            
            if idx != self._rcvd_idx:
                assert idx == self._rcvd_idx
            else:
                return self._process_data(data)

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        return data
    
    def _get_data(self):
        if self._load_local:
            while self._load_local_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                raise RuntimeError('load local thread exited unexpectedly')
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data
    
    def _try_get_data(self, timeout=_utils.worker.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout) # timeout is 5s
            return (True, data)
        except Exception as e:
            failed_workers = 0
            if self._worker_status and not self._worker.is_alive():
                self._shutdown_worker()
                failed_workers += 1
            if failed_workers > 0:
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(self._worker.pid))
            if isinstance(e, queue.Empty):
                print("data queue is empty")
                return (False, None)
            raise

    def _try_put_index(self):
        assert self._tasks_outstanding < self._num_elem_in_queue * self._num_workers
        for _ in range(self._num_workers):
            if self._worker_status:
                break
        else:
            # not found alive workers
            return
        self._index_queue.put(self._send_idx)
        self._tasks_outstanding += 1
        self._send_idx += 1
    
    def _shutdown_workers(self):
        if not self._shutdown:
            self._shutdown = True
            try:
                # Exit load_local_thread first.
                if hasattr(self, '_load_local_thread'):
                    self._load_local_thread_done_event.set()
                    self._worker_result_queue.put((None, None))
                    self._load_local_thread.join()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._worker_done_event.set()
                if self._worker_status:
                    self._shutdown_worker()
                self._worker.join()
                q = self._index_queue
                q.cancel_join_thread()
                q.close()
            finally:
                pass
    
    def _shutdown_worker(self):
        assert self._worker_status
        q = self._index_queue
        q.put(None)
        self._worker_status = False

    # def __del__(self):
    #     print("delete the DaemonProcessLoader instance")
    #     self._shutdown_workers()