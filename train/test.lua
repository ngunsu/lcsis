--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
-- Based on : https://github.com/torch/tutorials/tree/master/2_supervised
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'xlua'
require 'optim'

--------------------------------------------------------------------------------
-- Test
--------------------------------------------------------------------------------
function test()
    -- local vars
    local time = sys.clock()

    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()

    -- test over test data
    print '[INFO] Testing...'

    --New variables for error rate 95
    local labels = torch.Tensor(trainData:validation_size())
    local scores = torch.Tensor(trainData:validation_size())
    local scores_idx = 1

    for t = 1,trainData:validation_size() do
        -- disp progress
        xlua.progress(t, trainData:validation_size())
        -- get new sample
        local input = torch.Tensor(1, 2, 64, 64):float()
        input[{ {1},{},{},{} }] = trainData.validation_data[t]:clone():float():div(255)
        local p = input:view(1,2,64*64)
        p:add(-p:mean(3):expandAs(p))

        if opt.type == 'cuda' then input = input:cuda() end

        local target = trainData.validation_labels[t]

        if target == 1 then
            labels[t] = 1
        else
            labels[t] = 0
        end

        -- test sample
        local pred = model:forward(input)
        scores[t] = pred:float()
    end

    -- timing
    time = sys.clock() - time
    time = time / trainData:validation_size()
    print("[INFO] Time to test 1 sample = " .. (time*1000) .. 'ms')

    if opt.net == 'pseudo-siam-l2' then
        scores = scores*-1
    end
    --error_rate_at_95recall
    local error_rate = metrics.error_rate_at_95recall(labels, scores, true)
    test_er95 = error_rate
    print('[INFO] Error at 95%Recall: ' .. error_rate)
    print('\n\n')
end
