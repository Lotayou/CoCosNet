from options import DebugOptions, TrainOptions
from data import create_dataset
from model import create_model
from torch.backends import cudnn

#opt = DebugOptions()
opt = TrainOptions()
#os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids[0])  # test single GPU first

cudnn.enabled = True
cudnn.benchmark = True

loader = create_dataset(opt)
dataset_size = len(loader)
print('#training images = %d' % dataset_size)

net = create_model(opt)

for epoch in range(1,opt.niter+opt.niter_decay+1):
	print('Begin epoch %d' % epoch)
	for i, data_i in enumerate(loader):
		net.set_input(data_i)
		net.optimize_parameters()

		#### logging, visualizing, saving
		if i % opt.print_every == 0:
			net.log_loss(epoch, i)
		if i % opt.visual_every == 0:
			net.log_visual(epoch, i)

	net.save_networks('latest')
	if epoch % opt.save_every == 0:
		net.save_networks(epoch)
	net.update_learning_rate()
