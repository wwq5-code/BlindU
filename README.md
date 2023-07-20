# MCFU

# Server-side Masked Compressive Forgetting Unlearning (MCFU)  

### Overview
This repository is the official implementation of MCFU, and the corresponding paper is under review.


### Prerequisites

```
python = 3.8
pytorch = 1.4.0
matplotlib
numpy
```

We compress the whole project and add the MU_no_sahring_raw.zip file. 


### Running the experiments

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