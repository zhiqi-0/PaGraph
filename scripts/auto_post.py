import os
import re
import sys
import time
import socket
import argparse
import threading


datasets = {'reddit': ['reddit-small', 602],
            'wikitalk': ['wikitalk', 600],
            'livejournal': ['livejournal', 600],
            'lj-link': ['livejournallink-new', 600],
            'lj-large': ['lj-large', 400],
            'enwiki': ['enwiki', 400]}


def parseArgs():
    parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        	description='auto post server and trainer processes')
    parser.add_argument('--omp-thrs',       type = int, default = 2, 
                                            help = 'number of openmp threads for OMP_NUM_THREADS')
    parser.add_argument('--frame',          type = str, default = 'dgl', choices = ['dgl', 'pa'],
                                            help = 'framework, dgl or PaGraph')
    parser.add_argument('--model',          type = str, default = 'gcn', choices = ['gcn', 'gs'], 
                                            help = 'the training model, gcn and graphsage')
    parser.add_argument('--dataset',        type = str, default = 'reddit', choices = ['reddit', 'wikitalk', 'livejournal', 'lj-link', 'lj-large', 'enwiki'], 
                                            help = 'the training model, gcn and graphsage')
    parser.add_argument('--gpus',           type = str, default = '0', help = 'assign gpus')
    parser.add_argument('--preprocess',     action = 'store_true', help = 'preprocess or not')
    parser.add_argument('--remote-sample',  action = 'store_true', help = 'remote sampling or not')
    parser.add_argument('--scaling',        action = 'store_true', help = 'scaling evaluation')
    parser.add_argument('--pre-fetch',        action = 'store_true', help = 'enable prefetch for PaGraph')
    parser.add_argument('--home',           type = str, default = '/home/gpu/PaGraph', help = 'PaGraph home')
    return parser.parse_args()

class trainingModel:
    def __init__(self, model, args, resultpath):
        self.model = model
        self.path = resultpath
        self.args = args
        self.out_filename = None
    def server_process(self):
        pp = 'pp' if self.args.preprocess else 'nopp'
        sample = 'remote-sample' if self.args.remote_sample else 'local-sample'
        self.out_filename = "graph_%s_%s_%s.log"%(pp, sample, str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

        cmd = '''%(setomp)s python -u %(home)s/server/pa_server.py --dataset %(datapath)s --num-workers %(lenworker)d --feat-size %(fsize)d %(prep)s %(rsample)s 2>&1 | tee -a %(path)s/%(output)s'''%(
            {
                'setomp': '' if self.args.frame == 'dgl' and not self.args.scaling else 'OMP_NUM_THREADS=%d'%(self.args.omp_thrs),
                'home': self.args.home,
                'datapath': '/data/graphdata/%s'%(datasets[self.args.dataset][0]),
                'lenworker': len(self.args.gpus.split(',')),
                'fsize': datasets[self.args.dataset][1],
                'prep': '--preprocess' if self.args.preprocess else '',
                'rsample': '--sample' if self.args.remote_sample else '',
                'path': self.path,
                'output': self.out_filename
            })
        print(cmd)
        os.system(cmd)

    def trainer_process(self):
        cmd = '''%(setomp)s python -u %(home)s/prof/profile/%(frame)s_%(model)s.py --dataset %(datapath)s --gpu %(gpus)s --feat-size %(fsize)d %(prep)s %(rsample)s %(prefet)s 2>&1 | tee -a %(path)s/%(output)s'''%(
            {
                'setomp': '' if self.args.frame == 'dgl' and not self.args.scaling else 'OMP_NUM_THREADS=%d'%(self.args.omp_thrs),
                'home': self.args.home,
                'frame': self.args.frame,
                'model': self.args.model,
                'datapath': '/data/graphdata/%s'%(datasets[self.args.dataset][0]),
                'gpus': self.args.gpus,
                'fsize': datasets[self.args.dataset][1],
                'prep': '--preprocess' if self.args.preprocess else '',
                'rsample': '--remote-sample' if self.args.remote_sample else '',
                'prefet': '--pre-fetch' if self.args.pre_fetch else '',
                'path': self.path,
                'output': self.out_filename
            }
        )
        out_cmd = '''echo "%s" >> %s/%s'''%(cmd, self.path, self.out_filename)
        os.system(out_cmd)
        print(cmd)
        os.system(cmd)
    
    def detect_then_post_trainer(self):
        open_file = os.path.join(self.path, self.out_filename)
        while True:
            time.sleep(30)
            print("detecting:", self.out_filename)
            with open(open_file) as f:
                if "start running graph server on dataset" in f.read():
                    # post the trainer process
                    self.trainer_process()
                    break
        print("end of trainer process")

    def training(self):
        # post server deamon process
        thr = threading.Thread(target=self.server_process, args=())
        thr.setDaemon(True)
        thr.start()
        # periodically detect the output file and post the trainer process 
        self.detect_then_post_trainer()
        # wait the server process
        thr.join()
        self.cal_average_time()
        # cmd = "killall python"
        # os.system(cmd)
    
    def cal_average_time(self):
        open_file = os.path.join(self.path, self.out_filename)
        time_cost_per_epoch = list()
        with open(open_file) as f:
            for line in f:
                res = re.search('.*?average time:.*?(\d+\.\d*)', line)
                if res:
                    time_cost_per_epoch.append(float(res.group(1)))
        print(time_cost_per_epoch)
        
        with open(open_file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            avg_time = sum(time_cost_per_epoch)/len(time_cost_per_epoch)
            time_cost_per_epoch = [str(i) for i in time_cost_per_epoch]
            f.write(','.join(time_cost_per_epoch)+
                    '\n'+
                    str(avg_time)+
                    '\n'+
                    content)

if __name__ == "__main__":
    parsedArgs = parseArgs()
    
    dirname = str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    basepath = './trainResult/%s/%s/%s/%ddevice'%(parsedArgs.frame, parsedArgs.model, parsedArgs.dataset, len(parsedArgs.gpus.split(',')))
    os.system("mkdir -p %s"%(basepath))

    train = trainingModel(parsedArgs.model, parsedArgs, basepath)
    train.training()