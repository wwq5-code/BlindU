# MCFU

# Server-side Masked Compressive Forgetting Unlearning (MCFU)  

### Prerequisites

```
python = 3.8
pytorch = 1.4.0
matplotlib
numpy
```

### Running the experiments

1. To run the MCFU on MNIST
```
python /On_MNIST/temp_experiment_MNIST.py
```

2. To run the MCFU on CIFAR10
```
python /On_CIFAR/experiments_on_cifar.py
```

3. To run our reproduced and improved HBFU on MNIST
```
python /On_MNIST/FedHessian/backdoor_FedHessian2.py
```

4. To run our reproduced and improved HBFU on CIFAR
```
python /On_CIFAR/FedHessian/hessian_cifar_temp.py
```