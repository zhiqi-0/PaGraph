import os
import torch
import queue
from ..dataloader import SampleLoader
import time
from . import MP_STATUS_CHECK_INTERVAL, ExceptionWrapper

class ManagerWatchdog(object):
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False
    
    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead

class _IterableDatasetStopIteration(object):
    def __init__(self):
        self._result = '_IterableDatasetStopIteration'


def _worker_loop_init_sampler(loader, index_queue, data_queue, done_event, num_workers):
    try:
        # torch.set_num_threads(1)

        init_exception = None
        try:
            barrier_port = 9000+loader._epoch*100
            sampler = SampleLoader(loader._graph, loader._rank, loader._one2all, barrier_port, True)
        except Exception:
            init_exception = ExceptionWrapper('initializing the sampler')
            print("init the sampler error")
        
        iteration_end = False
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                index = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if index is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = next(sampler)
                except Exception as e:
                    if isinstance(e, StopIteration):
                        data = _IterableDatasetStopIteration()
                        iteration_end = True
                    else:
                        # unexpected exception
                        data = ExceptionWrapper('other exceptions')
            data_queue.put((index, data))
            del data, index
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()