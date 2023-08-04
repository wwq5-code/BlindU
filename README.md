# MCFU

# Server-side Masked Compressive Forgetting Unlearning (MCFU)  

### Overview
This repository is the official implementation of MCFU, and the corresponding paper is under review.

We compress the whole project and add the MU_no_sharing_raw.zip file for Artifact Evaluation. 


### Prerequisites

```
python = 3.10.10
torch==2.0.0
torchvision==0.15.1
matplotlib==3.7.1
numpy==1.23.5
```


### Running the experiment

1. To run the MCFU on MNIST
```
python /On_MNIST/temp_experiment_MNIST.py
```

2. To run the MCFU on CIFAR10
```
python /On_CIFAR/experiments_on_cifar.py
```

3. To run the MCFU on Adult
```
python /On_Adult/experiments_on_adult.py
```

4. To run our reproduced and improved HBFU on MNIST
```
python /On_MNIST/FedHessian/backdoor_FedHessian2.py
```

5. To run our reproduced and improved HBFU on CIFAR
```
python /On_CIFAR/FedHessian/hessian_cifar_temp.py
```

6. To run our reproduced and improved HBFU on Adult
```
python /On_Adult/FedHessian/BackdoorFedAvg2_temp.py
```


Here, we demonstrate the overall evaluations, which are also the main achievement claimed in the paper. We will explain the results and demonstrate how to achieve these results using the script and corresponding parameters.
Evaluated on 1080ti GPUs,

TABLE I: General Evaluation Results on MNIST, CIFAR10 and Adult:

On MNIST, EDR = 6%, \beta = 0.001, SR = 60%

| On MNIST             | Origin      | HBFU     |    VBU   |  MCFU_w  | MCFU_w/o |
| --------             | --------    | -------- | -------- | -------- | -------- |
| Mutual information   | 10.59       | 76.05    | 199.53   | 14.65    | 31.33    |
| Privacy leak attacks | 33.13 (MSE) | 22.78    | 0.00     | 42.91    | 31.22    |
| Backdoor Acc.        | 100%        | 5.06%    | 2.03%    | 1.19%    | 0.75%    |
| Acc. on test dataset | 97.35%      | 96.04%   | 88.19%   | 94.81%   | 94.78%   |
| Running time (s)     | 220         | 12.25    | 0.15     | 2.64     | 2.64     |

On CIFAR10, EDR = 6%, \beta = 0.01, SR = 60%

| On CIFAR10           | Origin      | HBFU     |    VBU   |  MCFU_w  | MCFU_w/o |
| --------             | --------    | -------- | -------- | -------- | -------- |
| Mutual information   | 3.02        | 6.93     | 117.65   | 5.37     | 7.19     |
| Privacy leak attacks | 74.63 (MSE) | 57.76    | 0.00     | 421.9    | 376.2    |
| Backdoor Acc.        | 99.90%      | 8.1%     | 7.9%     | 7.1%     | 6.9%     |
| Acc. on test dataset | 77.64%      | 75.85%   | 65.45%   | 76.03%   | 74.01%   |
| Running time (s)     | 1104        | 58.74    | 2.32     | 3.32     | 3.98     |


On Adult, EDR = 6%, \beta = 0.001, SR = 60%

| On Adult             | Origin       | HBFU     |    VBU   |  MCFU_w  | MCFU_w/o |
| --------             | --------     | -------- | -------- | -------- | -------- |
| Mutual information   | 2.14         | 3.66     | 10.99    | 3.48     | 2.74     |
| Privacy leak attacks | 90.76% (Acc.)| 96.80%   | 99.99%   | 57.68%   | 59.16%   |
| Backdoor Acc.        | 99.99%       | 99.99%   | 8.45%    | 9.41%    | 7.04%    |
| Acc. on test dataset | 85.57%       | 85.45%   | 64.64%   | 84.34%   | 84.05%   |
| Running time (s)     | 43.67        | 85.90    | 0.13     | 1.78     | 1.30     |



TABLE IV: Evaluation Results on MNIST, CIFAR10 and CIFAR100 using a P100 GPU:

On MNIST, EDR = 6%, \beta = 0.001, SR = 60%

| On MNIST             | Origin      | HBFU     |    VBU   |  MCFU_w  | MCFU_w/o |
| --------             | --------    | -------- | -------- | -------- | -------- |
| Mutual information   | 9.72        | 78.58    | 179.50   | 27.98    | 38.68    |
| Privacy leak attacks | 32.47 (MSE) | 21.82    | 0.00     | 37.99    | 27.41    |
| Backdoor Acc.        | 99.99%      | 6.61%    | 6.08%    | 3.80%    | 0.72%    |
| Acc. on test dataset | 96.91%      | 95.64%   | 93.89%   | 94.85%   | 94.52%   |
| Running time (s)     | 208.5       | 12.32    | 0.22     | 3.75     | 2.52     |

On CIFAR10, EDR = 6%, \beta = 0.01, SR = 60%

| On CIFAR10           | Origin      | HBFU     |    VBU   |  MCFU_w  | MCFU_w/o |
| --------             | --------    | -------- | -------- | -------- | -------- |
| Mutual information   | 2.42        | 6.42     | 89.39    | 5.19     | 5.77     |
| Privacy leak attacks | 58.70 (MSE) | 44.45    | 0.00     | 418.46   | 368.19   |
| Backdoor Acc.        | 100%        | 6.21%    | 7.03%    | 2.43%    | 1.26%    |
| Acc. on test dataset | 78.72%      | 76.85%   | 45.61%   | 77.04%   | 76.45%   |
| Running time (s)     | 997         | 53.04    | 3.29     | 4.78     | 4.18     |

On CIFAR100, EDR = 6%, \beta = 0.01, SR = 60%

| On CIFAR100          | Origin      | HBFU     |    VBU   |  MCFU_w  | MCFU_w/o |
| --------             | --------    | -------- | -------- | -------- | -------- |
| Mutual information   | 3.26        | 4.77     | 76.71    | 4.583    | 6.023    |
| Privacy leak attacks | 82.09 (MSE) | 48.00    | 0.00     | 428.85   | 391.49   |
| Backdoor Acc.        | 100%        | 7.61%    | 7.21%    | 6.08%    | 4.92%    |
| Acc. on test dataset | 52.24%      | 51.85%   | 30.82%   | 52.25%   | 52.13%   |
| Running time (s)     | 892         | 68.14    | 2.67     | 4.46     | 4.46     |
