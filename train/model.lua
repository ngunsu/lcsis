require 'nn'

model_manager = {}

function model_manager.build_2ch_net()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(2 ,96, 7,7, 3,3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.SpatialConvolution(96 ,192, 5,5, 1,1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.SpatialConvolution(192 ,256, 3,3, 1,1))
  model:add(nn.ReLU())
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(nn.Linear(256,1))
  return model
end

function model_manager.build_siam_net()
    rama = nn.Sequential()
    --rama:add(nn.Reshape(1,64,64))
    rama:add(nn.SpatialConvolution(1 ,96, 7,7, 3,3))
    rama:add(nn.ReLU())
    rama:add(nn.SpatialMaxPooling(2,2,2,2))
    rama:add(nn.SpatialConvolution(96 ,192, 5,5, 1,1))
    rama:add(nn.ReLU())
    rama:add(nn.SpatialMaxPooling(2,2,2,2))
    rama:add(nn.SpatialConvolution(192 ,256, 3,3, 1,1))
    rama:add(nn.ReLU())
    rama:add(nn.View(-1):setNumInputDims(3))

    siam_rama = nn.Parallel(2, 2)
    siam_rama:add(rama)
    siam_rama:add(rama:clone('weight','bias','gradWeight','gradBias'))

    model = nn.Sequential()
    model:add(siam_rama)
    model:add(nn.Linear(512,512))
    model:add(nn.ReLU())
    model:add(nn.Linear(512,1))

    return model
end

function model_manager.build_pseudo_siam_net()
    rama = nn.Sequential()
    --rama:add(nn.Reshape(1,64,64))
    rama:add(nn.SpatialConvolution(1 ,96, 7,7, 3,3))
    rama:add(nn.ReLU())
    rama:add(nn.SpatialMaxPooling(2,2,2,2))
    rama:add(nn.SpatialConvolution(96 ,192, 5,5, 1,1))
    rama:add(nn.ReLU())
    rama:add(nn.SpatialMaxPooling(2,2,2,2))
    rama:add(nn.SpatialConvolution(192 ,256, 3,3, 1,1))
    rama:add(nn.ReLU())
    rama:add(nn.View(-1):setNumInputDims(3))

    siam_rama = nn.Parallel(2, 2)
    siam_rama:add(rama)
    siam_rama:add(rama:clone())

    model = nn.Sequential()
    model:add(siam_rama)
    model:add(nn.Linear(512,512))
    model:add(nn.ReLU())
    model:add(nn.Linear(512,1))

    return model
end

