--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'nn'
require 'xlua'
require 'os'
require 'paths'
require 'image'
require './utils/metrics'
require 'pl.stringx'
require 'cutorch'
require 'cunn'
local cv = require 'cv'
require 'cv.features2d'

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
   cmd:option('-batch_mode', 1, 'Pass data through the network as batch')
   -- CPU/GPU related options
   cmd:option('-threads', 8, 'Number of threads')
   cmd:option('-type', 'float', 'Type: float | cuda')
   -- Dataset
   cmd:option('-dataset_path', '/opt/datasets/icip2015/', 'Dataset path')
   -- Models
   cmd:option('-model', './external/cvpr15networks/2ch/2ch_yosemite_nn.t7', 'Network model')
   cmd:option('-model_type', 'siam', 'Model type. Only used if matching strategy is l2')
   cmd:option('-matching_strategy', 'bf', 'Matching strategy bf=BruteForce l2=L2 distance (only siam|siam2stream)')
   cmd:text()
   -- Parse
   opt = cmd:parse(arg or {})
end

--------------------------------------------------------------------------------
-- Print options
--------------------------------------------------------------------------------

print('[INFO] Processing options')
print('[INFO] ---------------------------------')
print('[INFO] Using data type: ' .. opt.type)
print('[INFO] seed: ' .. opt.seed)
print('[INFO] batch_mode: ' .. opt.batch_mode)
print('[INFO] threads: ' .. opt.threads)
print('[INFO] dataset_path: ' .. opt.dataset_path)
print('[INFO] model: ' .. opt.model)
print('[INFO] matching strategy: ' .. opt.matching_strategy)
print('[INFO] ---------------------------------')

--------------------------------------------------------------------------------
-- Preparation for testing
--------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

-- Average precision
ap = torch.Tensor(44)

--------------------------------------------------------------------------------
-- Useful functions
--------------------------------------------------------------------------------
function file_to_table_kps(filename)
    local kps = {}
    local idx = 1
    print('[INFO] Loading... ' .. filename)
    for line in io.lines(filename) do
        splited_line = stringx.split(line, '\t')
        kps[idx] = {x=tonumber(splited_line[1])+1, y=tonumber(splited_line[2])+1, size=tonumber(splited_line[3])}
        idx = idx + 1
     end
    return kps
end

function file_to_table_gt_pairs(filename)
    local pairs_gt= {}
    local idx = 1
    print('[INFO] Loading... ' .. filename)
    for line in io.lines(filename) do
        splited_line = stringx.split(line, '\t')
        pairs_gt[idx] = {lwir_idx=tonumber(splited_line[5])+1, rgb_idx=tonumber(splited_line[6])+1}
        idx = idx + 1
    end
    return pairs_gt
end

function load_patches_from_table_kps(kps, im)
    local patches = torch.Tensor(#kps, 1, 64, 64)
    for j=1, #kps do
        if im:nDimension() == 2 then
            local patch = im[{{kps[j].y-31,kps[j].y+32}, {kps[j].x-31,kps[j].x+32}}]:clone()
            local p = patch:view(64*64)
            p:add(-p:mean(1):expandAs(p))
            patches[{{j},{1},{},{}}] = patch
        else
            local patch = im[{{1},{kps[j].y-31,kps[j].y+32}, {kps[j].x-31,kps[j].x+32}}]:clone()
            local p = patch:view(1,64*64)
            p:add(-p:mean(2):expandAs(p))
            patches[{{j},{},{},{}}] = patch
        end
    end
    return patches
end

function bf_matcher(lwir_patches, rgb_patches, net)
    local lwir_matches = {}
    local scores = torch.Tensor(lwir_patches:size(1)):fill(0)
    -- Compute descriptors
    for j=1, lwir_patches:size(1) do
        local max=-10000
        xlua.progress(j, lwir_patches:size(1))
        if opt.batch_mode == 1 then
            local ims_2ch =  torch.Tensor(rgb_patches:size(1), 2, 64, 64):float()
            for z = 1,rgb_patches:size(1) do
                ims_2ch[{{z},{1},{},{}}] = lwir_patches[j]
                ims_2ch[{{z},{2},{},{}}] = rgb_patches[z]
            end
            if opt.type == 'cuda' then
                ims_2ch = ims_2ch:cuda()
            end
            local out = net:forward(ims_2ch)
            local max_score, max_score_idx = torch.max(out, 1)
            scores[j] = max_score[1][1]
            lwir_matches[j] = max_score_idx[1][1]
        else
            local im_2ch =  torch.Tensor(2, 64, 64):float()
            best_idx = 1
            for z = 1,rgb_patches:size(1) do
                im_2ch[{ {1},{},{} }] = lwir_patches[j]
                im_2ch[{ {2},{},{} }] = rgb_patches[z]
                if opt.type == 'cuda' then
                    im_2ch = im_2ch:cuda()
                end
                local out = net:forward(im_2ch)
                if out:float()[1] > max then
                    best_idx = z
                    max = out:float()[1]
                end
            end
            scores[j] = max
            lwir_matches[j] = best_idx
        end
    end
    return lwir_matches, scores
end

function l2_matcher(lwir_patches, rgb_patches, net)
    local lwir_matches = {}
    local scores = torch.Tensor(lwir_patches:size(1)):fill(0)
    -- Compute descriptors
    local lwir_descriptors = net:forward(lwir_patches):clone()
    local rgb_descriptors = net:forward(rgb_patches):clone()
    local matcher = cv.BFMatcher{cv.NORM_L2, false}
    local matches = matcher:match{lwir_descriptors, rgb_descriptors}
    -- Store data
    for i=0, matches.size - 1 do
        lwir_matches[i+1] = matches.data[i].trainIdx + 1
        scores[i+1] = matches.data[i].distance
    end
     return lwir_matches, scores*-1
end


--------------------------------------------------------------------------------
--Process dataset
--------------------------------------------------------------------------------
for i =1, 44 do
    -- Load image pair
    local lwir_im_path = paths.concat(opt.dataset_path, 'lwir', 'lwir' .. i .. '.ppm')
    local rgb_im_path = paths.concat(opt.dataset_path, 'rgb', 'rgb' .. i .. '.ppm')
    print('[INFO] Loading... ' .. lwir_im_path)
    local lwir_im = image.load(lwir_im_path,1)
    print('[INFO] Loading... ' .. rgb_im_path)
    local rgb_im = image.load(rgb_im_path,1)

    local lwir_kp_path = paths.concat(opt.dataset_path, 'gt', 'gt', 'lwir_kps','lwir' .. i .. '.kps')
    local rgb_kp_path = paths.concat(opt.dataset_path, 'gt', 'gt', 'rgb_kps', 'rgb' .. i .. '.kps')
    local pairs_gt_path = paths.concat(opt.dataset_path, 'gt', 'gt', 'pairs', 'lwir' .. i .. '_' ..'rgb' .. i .. '.gt')

    local lwir_kps = file_to_table_kps(lwir_kp_path)
    local rgb_kps = file_to_table_kps(rgb_kp_path)
    local pairs_gt = file_to_table_gt_pairs(pairs_gt_path)

    -- Load patches
    local lwir_patches = load_patches_from_table_kps(lwir_kps, lwir_im)
    local rgb_patches = load_patches_from_table_kps(rgb_kps, rgb_im)

    -- Load Network
    local net = torch.load(opt.model)
    if opt.type == 'cuda' then
        net = net:cuda()
    elseif opt.type == 'float' then
        net = net:float()
    end
    net:evaluate()

    -- Match patches (Keypoints)
    local lwir_matches, scores
    if opt.matching_strategy=='bf' then
        lwir_matches,scores = bf_matcher(lwir_patches, rgb_patches, net)
    elseif opt.matching_strategy=='l2' then
        if opt.model_type == 'siam' then
            local siam_net = net:get(1):get(1)
            lwir_matches,scores = l2_matcher(lwir_patches, rgb_patches, siam_net)
        end
    end
    -- Compare with GT
    local labels = torch.Tensor(#lwir_matches):fill(0)
    for j=1,#lwir_matches do
        for z=1, #pairs_gt do
            if lwir_matches[j] == pairs_gt[z].rgb_idx and pairs_gt[z].lwir_idx == j then
                labels[j] = 1
                break
            end
        end
    end

 ap[i] = metrics.average_precision{labels=labels, scores=scores, descending=true}
 print('AP ' .. i .. ': ' .. ap[i])

end
print('MAP: ' .. ap:mean())
