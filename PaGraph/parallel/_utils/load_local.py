import os
import torch
import queue
from . import MP_STATUS_CHECK_INTERVAL

def _load_local_loop(in_queue, out_queue, device_id, done_event):
    torch.set_num_threads(1)
    torch.cuda.set_device(device_id)

    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue

        idx, data = r
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        
        del r # save memory