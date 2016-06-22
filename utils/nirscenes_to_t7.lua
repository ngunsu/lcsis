---------------------------------------------------------------------------------------------------
-- This script generates image patches using the nirscenes dataset. It generates 9 files
-- country.t7 ... water.t7
-- It requires:
-- 1.- Nirscenes dataset (uncompressed): http://ivrl.epfl.ch/supplementary_material/cvpr11/
--    *It is necessary to convert the .tiff images to ppm (Torch doesn't support tiff)
-- 2.- CSV folder with patches information: https://github.com/ngunsu/lcsis
--
-- Lua/Torch libs required and not in the standard torch installation
-- 1.- csvigo (to install) -> luarocks install csvigo
---------------------------------------------------------------------------------------------------
-- Cristhian Aguilera

---------------------------------------------------------------------------------------------------
-- Required libs
---------------------------------------------------------------------------------------------------
torch = require 'torch'
paths = require 'paths'
csvigo = require 'csvigo'
image = require 'image'

---------------------------------------------------------------------------------------------------
-- Nirscenes sequences
---------------------------------------------------------------------------------------------------
sequences = {'country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'}

---------------------------------------------------------------------------------------------------
-- Argument parsing
---------------------------------------------------------------------------------------------------
-- Info message
cmd = torch.CmdLine()
cmd:text()
cmd:text('Nirscenes to t7 dataset')
cmd:text()
cmd:text('Options:')

-- Dataset Options
cmd:option('-dataset_path', '/opt/datasets/nirscenes/', 'Dataset path')
cmd:option('-csv_path', '../datasets/nirscenes/csv/', 'CSV folder with patches information')
cmd:option('-output_folder', '../datasets/nirscenes/', 'Output folder')
cmd:text()

-- Parse options
opt = cmd:parse(arg or {})

---------------------------------------------------------------------------------------------------
-- Processing dataset
---------------------------------------------------------------------------------------------------
-- Set default tensor to ByteTensor to save storage
torch.setdefaulttensortype('torch.ByteTensor')

-- For each sequence
for __,s in pairs(sequences) do
    print('Processing ' .. s .. '...')

    -- Read csv file
    local s_csv_path = paths.concat(opt.csv_path, s .. '.csv')
    local s_csv = csvigo.load(s_csv_path)

    -- Sequence path
    local s_path = paths.concat(opt.dataset_path, s)

    -- Num of patches in the csv file
    local n_patches = #s_csv.rgb

    -- Prepare torch tensor
    local data = torch.Tensor(n_patches, 2, 64, 64)
    local labels = torch.Tensor(n_patches)

    -- For every patch
    local rgb_image_name = nil
    local nir_image_name = nil
    local last_rgb_image_name = ''
    local last_nir_image_name = ''
    local rgb_image = nil
    local nir_image = nil

    for i=1, n_patches do
        -- Image names
        rgb_image_name = s_csv.rgb[i]
        nir_image_name = s_csv.nir[i]
        local rgb_path = paths.concat(s_path,rgb_image_name)
        local nir_path = paths.concat(s_path,nir_image_name)
        if rgb_image_name ~= last_rgb_image_name  then
            rgb_image = image.load(rgb_path, 1, 'byte')
            nir_image = image.load(nir_path, 1, 'byte')
            last_rgb_image_name = rgb_image_name
        end
        local patch_type = s_csv.type[i]
        local rgb_x = s_csv.rgb_x[i]
        local rgb_y = s_csv.rgb_y[i]
        local nir_x = s_csv.nir_x[i]
        local nir_y = s_csv.nir_y[i]
        -- Extract patch
        data[{ {i},{1},{},{} }] = rgb_image[{ {rgb_y-31,rgb_y+32},{rgb_x-31,rgb_x+32} }]
        data[{ {i},{2},{},{} }] = nir_image[{ {nir_y-31,nir_y+32},{nir_x-31,nir_x+32} }]
        if patch_type == 'positive' then
            labels[i] = 1
        else
            labels[i] = 0
        end
    end

    -- Store data in the current folder
    local set = {}
    set.data = data
    set.labels = labels
    torch.save(paths.concat(opt.output_folder,s .. '.t7'), set, 'ascii')
end

