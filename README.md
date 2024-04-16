# MTL-model
Five P450s enzyme substrate prediction platform, including: CYP1A2, 2C9, 2C19, 2D6, and 3A4.
## Installation
Before you run the code, we are recommended you to create a conda environment for example:

```conda create -n MTL-model```

Then configure the relevant installation packages:

```conda env create -f environment.yaml ```

Finally activate the environment:

```conda activate MTL-model```

## Usage
The input Data file, in the format Data\MTL_data, contains a multi-tasking dataset of 10 seeds for retraining the model:

``` python train.py ```

Alternatively, you can directly enter a csv file containing SMILES and use the trained model to get predictions:

```python external_predict.py```

![工作流程](https://github.com/fang-jiaoj/MTL-model/blob/main/%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.JPG)
