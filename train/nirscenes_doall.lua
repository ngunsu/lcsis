--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
-- Based on : https://github.com/torch/tutorials/tree/master/2_supervised
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require '../utils/nirscenes_data.lua'
require '../utils/metrics.lua'
require 'pl.stringx' -- luarocks install penlight
require 'model'
require 'optim'

--------------------------------------------------------------------------------
-- Command line arguments
--------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Cross-spectral Training')
cmd:text()
cmd:text('Options:')

-- Global
cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')

-- CPU/GPU related options
cmd:option('-threads', 8, 'Number of threads')
cmd:option('-type', 'float', 'Type: float | cuda')
cmd:option('-gc', 20 , 'Garbage collector every gc batches')

-- Logs
cmd:option('-save', 'results', 'Subdirectory to save/log experiments in')

-- Dataset
cmd:option('-dataset_path', '../datasets/nirscenes/', 'Dataset path')
cmd:option('-training_sequences', 'country', 'Training sequences. One: country or multiples: country_forest')

-- Training
cmd:option('-net','2ch','Net type: 2ch | siam | psiam ')
cmd:option('-data_augmentation', 0 , 'Data augmentation: 0 = no data augmentation 1 = data augmentation')
cmd:option('-criterion', 'MarginCriterion', 'Loss criterion')
cmd:option('-batchSize', 256, 'Mini-batch size (1 = pure stochastic)')
cmd:option('-maxIter', 25, 'Maximum num of iterations')
cmd:option('-learningRate', 5e-2, 'learning rate at t=0')
cmd:option('-learningRateDecay', 0 , 'Learning rate decay')
cmd:option('-weightDecay', 0.0005, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:text()

opt = cmd:parse(arg or {})


--------------------------------------------------------------------------------
-- Print options
--------------------------------------------------------------------------------

print('[INFO] Processing options')
print('[INFO] ---------------------------------')
print(opt)
print('[INFO] ---------------------------------')

--------------------------------------------------------------------------------
-- Set global configuration
--------------------------------------------------------------------------------
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

--------------------------------------------------------------------------------
-- Load training and testing dataset
--------------------------------------------------------------------------------
training_sequences = string.split(opt.training_sequences, '_')
trainData = nirscenes_patches.load_dataset(opt.dataset_path, training_sequences, opt.data_augmentation)
print('[INFO] Training dataset size #: ' .. trainData:size())
print('[INFO] Training dataset labels size #: ' .. trainData.labels:size(1))
print('[INFO] Validation dataset size #: ' .. trainData:validation_size())
print('[INFO] Validation dataset labels size #: ' .. trainData.validation_labels:size(1))
collectgarbage()
collectgarbage()


--------------------------------------------------------------------------------
-- Make preparations for training
--------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
   print('[INFO] Loading cuda libs')
   require 'cunn'
   require 'cudnn'
end

--------------------------------------------------------------------------------
-- Build model
--------------------------------------------------------------------------------
print('[INFO] Preparing network')
if opt.net == '2ch' then
    model = model_manager.build_2ch_net()
elseif opt.net == 'siam' then
    model = model_manager.build_siam_net()
elseif opt.net == 'psiam' then
    model = model_manager.build_pseudo_siam_net()
end

--------------------------------------------------------------------------------
-- Define Loss
--------------------------------------------------------------------------------

print('[INFO] Defining loss')
margin = 1
criterion = nn.MarginCriterion(margin)

--------------------------------------------------------------------------------
-- Load training and test functions
--------------------------------------------------------------------------------
dofile 'train.lua'
dofile 'test.lua'

--------------------------------------------------------------------------------
-- Create Logger
--------------------------------------------------------------------------------
loss_logger = optim.Logger(paths.concat(opt.save, 'loss_logger.log'))
loss_curve_logger = optim.Logger(paths.concat(opt.save, 'loss_curve_logger.log'))
accuracy_logger = optim.Logger(paths.concat(opt.save, 'accuracy_logger.log'))
loss_error = 0
train_er95 = 0
val_er95 = 0

--------------------------------------------------------------------------------
-- Train dataset
--------------------------------------------------------------------------------
print('[INFO] Start training dataset...\n')
for i=1, opt.maxIter do
    loss_error = 0
    train_er95 = 0
    val_er95 = 0
    train()
    test()
    loss_logger:add{['loss'] = loss_error}
    accuracy_logger:add{['training error'] = train_er95,['test error'] = test_er95}
end

loss_logger:style{['loss'] = '-'}
loss_logger:plot()
loss_curve_logger:style{['loss_curve'] = '-'}
loss_curve_logger:plot()
accuracy_logger:style{['training error'] = '-', ['test error'] = '-'}
accuracy_logger:plot()


print('[INFO] End')
