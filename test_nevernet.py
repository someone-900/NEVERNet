from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import argparse

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id=0
opt.verbose = False

datadir = '/home/spark/NEVERNet/data/input'   # folder where Gradio saves uploaded images
result_dir = '/home/spark/NEVERNet/data/output'  # folder where results should be saved

eval_dataset_real = datasets.RealDataset(datadir)
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True
)

engine = Engine(opt)
res = engine.test(eval_dataloader_real, savedir=result_dir)
