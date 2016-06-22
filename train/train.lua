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
require 'math'

--------------------------------------------------------------------------------
-- Cuda
--------------------------------------------------------------------------------
if opt.type == 'cuda' then
    require 'cudnn'
    model:cuda()
    cudnn.convert(model, cudnn)
    criterion:cuda()
end

--------------------------------------------------------------------------------
-- Prepare for training
--------------------------------------------------------------------------------
if model then
    parameters,gradParameters = model:getParameters()
end

-- Set SGD parameters
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
}
optimMethod = optim.sgd

print('[INFO] Optimization configuration')
print(optimState)

-------------------------------------------------------------------------------
-- Train
--------------------------------------------------------------------------------
function train()
    -- Epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ
    -- in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(trainData:size())

    -- do one epoch
    print('[INFO] ==============  Start training epoch #' .. epoch .. ' ==============')
    print('[INFO]' .. ' batchSize: ' ..  opt.batchSize )

    --New variables for error rate 95
    local labels = torch.Tensor(trainData:size(1))
    local scores = torch.Tensor(trainData:size(1))
    local scores_idx = 1

    for t = 1,trainData:size(),opt.batchSize do
        -- Clean memory just in case
        if math.floor(math.fmod(t/opt.batchSize, opt.gc)) == 0 then
            collectgarbage()
            collectgarbage()
        end
        -- disp progress
        xlua.progress(t, trainData:size())

        -- create mini batch
        local batchsize = math.min( t + opt.batchSize-1,trainData:size()) - t + 1
        local inputs = torch.Tensor(batchsize, 2, 64, 64):float()
        local targets = torch.Tensor(batchsize):float()
        local index = 1

        for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
            -- load new sample
            input = torch.Tensor(1,2,64,64)
            input[{ {1},{},{},{} }] = trainData.data[shuffle[i]]:clone():float():div(255)

            -- Normalize path
            local p = input:view(1,2,64*64)
            p:add(-p:mean(3):expandAs(p))

            -- Store patch in batch and store target (1,-1)
            inputs[{ {index},{},{},{} }] = input[{ {1},{},{},{} }]:clone()
            if trainData.labels[shuffle[i]] == 1 then
                targets[index] = 1
            else
                targets[index] = -1
            end
            index = index + 1
        end

        if opt.type == 'cuda' then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- Create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- Get new parameters
            if x ~= parameters then
                parameters:copy(x)
             end

            -- Reset gradients
            gradParameters:zero()

            local outputs = nil
            local outputs = model:forward(inputs)
            local err = criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            -- Compute train performance
            scores[{ {scores_idx,scores_idx+outputs:clone():float():size(1)-1} }] = outputs:clone():float()
            labels[{ {scores_idx,scores_idx+outputs:clone():float():size(1)-1} }] = targets:clone():float()
            scores_idx = scores_idx + outputs:clone():float():size(1)

            loss_curve_logger:add{['loss_curve'] = err }
            loss_error = loss_error + err
            return err ,gradParameters
        end

        -- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
    end
    local error_rate = metrics.error_rate_at_95recall(labels, scores, true)
    train_er95 = error_rate
    print('\n[INFO] loss error: ' .. loss_error)
    print('\n[INFO] Error at 95%Recall: ' .. error_rate)

    -- Elapsed time
    time = sys.clock() - time
    time = time / trainData:size()
    print("[INFO] Time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- save/log current net
    local filename = paths.concat(opt.save, 'model_epoch' .. epoch .. '.net')
    --local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('[INFO] Saving model to '..filename)
    net_to_save = model:clone()
    if opt.type == 'cuda' then
        cudnn.convert(net_to_save, nn)
        net_to_save = net_to_save:float()
    end
    net_to_save = net_to_save:clearState()
    torch.save(filename, net_to_save)
    epoch = epoch + 1

end
