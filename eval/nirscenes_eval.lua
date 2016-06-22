--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'math'
require 'nn'
require 'xlua'
require 'paths'
require '../utils/nirscenes_data.lua'
require '../utils/metrics.lua'

--------------------------------------------------------------------------------
-- Command line arguments
--------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('NIRSCENES-patch evaluation')
cmd:text()
cmd:text('Options:')

-- Global
cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')

-- CPU/GPU related options
cmd:option('-threads', 8, 'Number of threads')
cmd:option('-type', 'cuda', 'Type: float | cuda')

-- Sequence path
cmd:option('-seq_path', '../datasets/nirscenes/country.t7', 'Sequence t7 filepath')

-- Models
cmd:option('-net', '../trained_networks/2ch_country.t7', 'Trained network')
cmd:option('-net_type', '2ch', 'Network type [2ch|siam|psiam]')
cmd:text()

-- Parse
opt = cmd:parse(arg or {})

--------------------------------------------------------------------------------
-- Print options
--------------------------------------------------------------------------------

print('[INFO] Processing options')
print(opt)
print('[INFO] ---------------------------------')

--------------------------------------------------------------------------------
-- Set global configuration
--------------------------------------------------------------------------------
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

--------------------------------------------------------------------------------
-- Load test data
--------------------------------------------------------------------------------
print('[INFO] Loading ' .. opt.seq_path .. ' ...')
test_data = torch.load(opt.seq_path, 'ascii')
print('[INFO] Testdata size #: ' .. test_data.data:size(1))

--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------
print('[INFO] Loading network ...')
model = torch.load(opt.net)
if opt.type == 'cuda' then
    require 'cunn'
    require 'cutorch'
    require 'cudnn'
    model:cuda()
    cudnn.convert(model, cudnn)
elseif opt.type == 'float' then
    model:float()
end

-------------------------------------------------------------------------------
-- Pass data through the net
---------------------------------------------------------------------------------
print('[INFO] Testing ...')
torch.setdefaulttensortype('torch.FloatTensor')

-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
model:evaluate()

-- Prepare Tensor to store prediction scores
scores = torch.Tensor(test_data.data:size(1))
if opt.type == 'cuda' then
    scores = scores:cuda()
end

-- Evaluate
for i=1, test_data.data:size(1) do
    d,r = math.modf(i/1000)
    if r==0 then
        xlua.progress(i,test_data.data:size(1))
    end
    if opt.net_type == '2ch' then
        local input = test_data.data[i]:clone():float():div(255)
        -- Normalize the patch
        local p = input:view(2, 64*64)
        p:add(-p:mean(2):expandAs(p))
        -- Prepare tyhe input for cuda
        if opt.type == 'cuda' then
            input= input:float():cuda()
        end
        -- Predict
        scores[i] = model:forward(input)
    else
        local input = torch.Tensor(1,2,64,64)
        input[{ {1},{},{},{} }] = test_data.data[i]:clone():float():div(255)
        -- Normalize the patch
        local p = input:view(1, 2, 64*64)
        p:add(-p:mean(3):expandAs(p))
        -- Prepare tyhe input for cuda
        if opt.type == 'cuda' then
            input= input:float():cuda()
        end
        -- Predict
        scores[i] = model:forward(input)
    end
end

--------------------------------------------------------------------------------
-- Compute error at 95% Recall
--------------------------------------------------------------------------------
print('[INFO] Computing error ...')

scores = scores:float()
error =  metrics.error_rate_at_95recall(test_data.labels, scores, true)
print('[INFO] Error at 95% Recall:' .. error)
