--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'xlua'
require 'os'
require 'paths'
require 'image'
require '../utils/metrics.lua'

--------------------------------------------------------------------------------
-- Parse command line arguments
--------------------------------------------------------------------------------
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('LWIR_VIS local features evaluation')
   cmd:text()
   cmd:text('Options:')
   -- Global
   cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')

   -- Dataset
   cmd:option('-dataset_path', '../datasets/icip2015/', 'Dataset path')
   -- Models
   cmd:option('-net', '../trained_networks/2ch_country.t7', 'Trained network')
   cmd:text()
   -- Parse
   opt = cmd:parse(arg or {})
end

--------------------------------------------------------------------------------
-- Print options
--------------------------------------------------------------------------------

print(opt)

--------------------------------------------------------------------------------
-- Preparation for testing
--------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

-- Load the data
data = torch.load(paths.concat(opt.dataset_path, 'icip2015eval.t7'))

--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------
print('Loading model ...')
net = torch.load(opt.net)
cudnn.convert(net, cudnn)
net:cuda()
print(net)
net:evaluate()

--------------------------------------------------------------------------------
--Process dataset
--------------------------------------------------------------------------------
-- Average precision
ap = torch.Tensor(44)

for i =1, 44 do
    print('Processing pair # ' .. i)
    local lwir_patches = data[i].lwir_patches
    local rgb_patches = data[i].rgb_patches
    local gt_pairs_lwir_rgb = data[i].gt_pairs_lwir_rgb

    -- Prepare vars for results
    local scores = torch.Tensor(lwir_patches:size(1)):fill(0)
    local lwir_matches = {}

    -- Normalize data
    lwir_patches = lwir_patches:float()
    rgb_patches = rgb_patches:float()
    local size_patch=64
    lwir_patches:div(255)
    rgb_patches:div(255)
    for j=1, lwir_patches:size(1) do
        lwir_patches[j]:add(-lwir_patches[j]:mean())
    end
    for j=1, rgb_patches:size(1) do
        rgb_patches[j]:add(-rgb_patches[j]:mean())
    end

    -- For each lwir patch
    local tmp_patches = torch.Tensor(rgb_patches:size(1),2,size_patch,size_patch):float()
    tmp_patches[{ {},{2},{},{} }] = rgb_patches:float():clone()

    for j =1, lwir_patches:size(1) do
        xlua.progress(j, lwir_patches:size(1))

        local r_lwir = torch.repeatTensor(lwir_patches[j], rgb_patches:size(1),1,1)
        tmp_patches[{ {},{1},{},{} }] = r_lwir
        local j_tmp_patches = tmp_patches:clone()

        j_tmp_patches= j_tmp_patches:cuda()

        local out = net:forward(j_tmp_patches):clone():float()
        local max_val, max_idx = out:max(1)
        scores[j] = max_val[1]
        lwir_matches[j] = max_idx[1][1]
    end

    -- Compare with gt
    local labels = torch.Tensor(#lwir_matches):fill(0)
    for j=1,#lwir_matches do
        if gt_pairs_lwir_rgb[j] ~= nil then
            if gt_pairs_lwir_rgb[j] == lwir_matches[j] then
                labels[j] = 1
            end
        end
    end

    descending = true

    ap[i] = metrics.average_precision{labels=labels, scores=scores, descending=descending}
    print('AP ' .. i .. ': ' .. ap[i])
end
print('MAP: ' .. ap:mean())
