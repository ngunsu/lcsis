# Learning cross-spectral similarity measures with deep convolutional neural networks

[PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w9/papers/Aguilera_Learning_Cross-Spectral_Similarity_CVPR_2016_paper.pdf) | [webpage](http://www.crisale.net/publication/cvprw16/)

Bibtex
```latex
@inproceedings{Aguilera_cvprw_2016,
    organization = { IEEE  },
    year = { 2016  },
    pages = { 9  },
    month = { Jun  },
    location = { Las vegas, USA  },
    booktitle = { The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops  },
    author = { Cristhian A. Aguilera and Francisco J. Aguilera and Angel D. Sappa and Cristhian Aguilera and Ricardo Toledo  },
    title = { Learning cross-spectral similarity measures with deep convolutional neural networks  },
}
```

## Instructions

First install the torch framework and cudnn

1. [Install torch](http://torch.ch/docs/getting-started.html#_)
2. [Cudnn torch](https://github.com/soumith/cudnn.torch)

### Datasets

#### Nirscenes patches

Follow the following steps to generate the dataset

1. Download gt csv

    ```bash
    cd datasets
    ./download_nirscenes_csv.sh
    ```

2. Download the original nirscenes dataset [link](http://ivrl.epfl.ch/supplementary_material/cvpr11/)
3. Decompress the dataset on /datasets/nirscenes
4. Convert the images in each folder to ppm  (Torch doesn't support tiff). Use your favorite software
5. Install csvigo

    ```bash
    luarocks install csvigo
    ```

6. Use our script to generate the dataset

    ```bash
    cd utils
    th nirscenes_to_t7.lua
    ```

The t7 dataset is stored in datasets/nirscenes


#### VIS-LWIR ICIP2015

1. Download the dataset

```bash
cd datasets
./download_icip_dataset.sh
```

It's easier if you are the owner of the dataset :)

### Eval

#### Nirscenes eval (cpu and cuda support)

Evaluation code can be found in the *eval* folder. To eval one sequence:

1. You have to generate the nirscenes patch dataset
2. Install xlua

    ```bash
    luarocks install xlua
    ```

3. Run

    ```bash
    cd eval
    th nirscenes_eval.lua -seq_path ../dataset/nirscenes/[sequence].t7 -net .. [trained network]
    ```

For example, to eval the field sequence using the 2ch_country network
    ```bash
    th nirscenes_eval.lua -seq_path ../dataset/nirscenes/field.t7 -net ../trained_networks/2ch_country.net -net_type 2ch
    ```

For more options, run 
    ```bash
    th nirscenes_eval -h
    ```

#### VIS-LWIR eval (ICIP2015) (just cuda support)

1. You have to download the dataset first
2. Run

    ```bash
    cd eval
    th icip2015_eval.lua -dataset_path ../nirscenes/icip2015/ -net [trained network]
    ```

For example. To eval 2ch_country

    ```bash
    cd eval
    th icip2015_eval.lua -dataset_path ../nirscenes/icip2015/ -net ../trained_networks/2ch_country.t7
    ```

### Training

1. Install penlight

    ```bash
    luarocks install penlight
    ```

2. Train a network

    ```bash
    cd train
    th nirscenes_doall.lua -training_sequences [country|field|...] -net [2ch|siam|psiam]
    ```

For example, train a 2ch network using the country sequence

 ```bash
 cd train
 th nirscenes_doall.lua -training_sequences country -net 2ch
 ```

Results will be stored in the results folder.For more options, run

```bash
th nirscenes_doall.lua -h
```

*Note* The training code is different from the one used in the article. This new version is much faster. 


