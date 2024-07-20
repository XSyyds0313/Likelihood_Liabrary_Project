# Transformer-based models versus xLSTM for orderbook data

This repository is the implementation for the Project: Transformer-based models versus xLSTM for orderbook data.

## Implementation details & References

The Implmentation of this project is based on a few open-source repositories. I declared them here and thanks for their valueable works.

This repository is built on the code base of Autoformer.&#x20;

The implementation of Autoformer, Informer, Transformer is from:

[https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer "https://github.com/thuml/Autoformer")

The implementation of FEDformer is from:

[https://github.com/MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer "https://github.com/MAZiqing/FEDformer")

The implementation of NS_Transformer, NS_Informer, NS_Autoformer, is from:

[https://github.com/thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers "https://github.com/thuml/Nonstationary_Transformers")

The implementation of iTransformer, iInformer is from:

[https://github.com/thuml/iTransformer](https://github.com/thuml/iTransformer "https://github.com/thuml/iTransformer")


# To get start

## Enviroment

It is recommended to run the code in a virtual environment. After initializing the virtual environment, install the requirement by:

```bash
pip install -r requirements.txt
```

## Tasks


## Dataset


## Training and Testing

If you want to save time for training and just testing the result, set the args "--is_training_and_testing" to [0,1]. To reproduce the result, experiment scripts are provided under `./scripts` directory. The scripts are written in shell language.  Make sure to run them in Linux or use the wsl subsystem in Windows. For example, now we want to run the task4 by using Transformer model, the example of running the script is provided below:

```bash
chmod +x scripts/task4_script/Transformer.sh
scripts/task4_script/Transformer.sh

```

Once starting training the model, a log file will be saved under the `./logs` directory.


## <a name="other"></a> Other Files Description

`run.py`: The main python file executed by the shell script; you can define different parameters in this file and directly run `python run.py` to start the training/validation/testing process.

`data_provider/data_loader.py` : Change this file if you want to customize and use your own dataset.

`exp/exp_main.py` : This file controls the training/validation/testing logic.

`layers`: Layers and components of Transformer-based models and xLSTM.

`ns_layers`: Layers and components of Non-Stationary Transformer.

`i_models`: Components of iTransformer models.

`data_peprocession.ipynb`: This notebook is used for data preprocessing and parameter tuning. 

`construct_debug_data.ipynb`: This notebook is used to construct data slice for debugging. 

`utils/backtest_functions.py`: Functions for backtest in train and test set

`utils/helper_functions.py`: Load and save of PKL files, parallel computing, and calculation of backtesting indicators

`utils/product_info.py`: Price difference between buying and selling, handling fees and rates for each product

`utils/tasks.py`: Reshape model outputs for each task
# model lightweight
InfoBatch-ICLR2024
https://github.com/henryqin1997/InfoBatch






