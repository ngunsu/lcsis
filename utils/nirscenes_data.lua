--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'paths'
require 'image'
require 'math'

--------------------------------------------------------------------------------
-- Nirscenes manager
--------------------------------------------------------------------------------
nirscenes_patches = {}

--------------------------------------------------------------------------------
-- Load sequences
--------------------------------------------------------------------------------
function nirscenes_patches.load_sequence(dataset_path, sequence)
    print('[INFO] Loading ' .. sequence .. ' sequence...')
    -- Load sequence
    local seq_path = paths.concat(dataset_path, sequence .. '.t7')
    local seq_data = torch.load(seq_path, 'ascii')
    return seq_data
end

--------------------------------------------------------------------------------
-- Load dataset
--------------------------------------------------------------------------------
function nirscenes_patches.load_dataset(dataset_path, sequences, data_augmentation)
    local patches_data = {}
    local labels = torch.Tensor():float()
    local idx = 1
    for key, sequence in pairs(sequences) do
        local seq = nirscenes_patches.load_sequence(dataset_path, sequence)
        for i=1, seq.data:size(1) do
            patches_data[idx] = seq.data[i]
            idx = idx + 1
        end
        if labels:nDimension() == 0 then
            labels = seq.labels:clone()
        else
            labels = torch.cat(labels, seq.labels:clone())
        end
    end

    -- Augmenting data
    local original_size = #patches_data
    local original_labels = labels:clone()
    if data_augmentation == 1 then
        -- Vertical flip
        for i=1, original_size do
            table.insert(patches_data, image.vflip(patches_data[i]:clone()) )
        end
        labels = torch.cat(labels,original_labels)

        -- Horizontal flip
        for i=1, original_size do
            table.insert(patches_data, image.hflip(patches_data[i]:clone()) )
        end
        labels = torch.cat(labels,original_labels)

        -- Rotate 90
        for i=1, original_size do
            table.insert(patches_data, image.rotate(patches_data[i]:clone(),90) )
        end
        labels = torch.cat(labels,original_labels)
    end

    -- Divide in training and validation
    local shuffle = torch.randperm(#patches_data)
    local training_samples = math.floor(#patches_data * 0.8)
    local training_data = {}
    local training_labels = torch.Tensor(training_samples):float()
    for i=1, training_samples do
        training_data[i] = patches_data[shuffle[i]]:clone()
        training_labels[i] = labels[shuffle[i]]
    end
    local validation_data = {}
    local validation_labels = torch.Tensor(#patches_data-training_samples)
    for i = training_samples+1, #patches_data do
        validation_data[i-training_samples] = patches_data[shuffle[i]]:clone()
        validation_labels[i-training_samples] = labels[shuffle[i]]
    end

    -- Preparing dataset
    local dataset = {}
    dataset.data = training_data
    dataset.labels = training_labels
    dataset.validation_data = validation_data
    dataset.validation_labels = validation_labels

    -- Adding size function
    function dataset:size()
        return #training_data
    end
    function dataset:validation_size()
        return #patches_data-#training_data
    end
    return dataset
end

