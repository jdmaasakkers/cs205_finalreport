# Parallelized Image Recognition in OpenMP + MPI and Spark
## CS205 - Final Report

Image recognition is a hot topic in machine learning. It has a lot of applications including in tagging people in photos, augmented reality, improving image search, and self-driving vehicles that have to identify different objects on the road and respond based on their location and movement. In this project statistical learning theory in used in the OpenMP+MPI and Spark frameworks to classify images. Both model and data parallelization are explored as many examples of labeled images are critical to training an accurate model.

(%% Include state-of-the-art image recognition including references ~ existing work on the problem - Dan)

### Learning algorithm
A multi-class linear classifier (one hidden layer neutral network) is implemented to perform a training, validation, and testing split on the data. The current implementation uses the MNIST database [(LeCun et al. 1998)](http://yann.lecun.com/exdb/mnist/) that consists handwritten digits (0-9). The database includes a training set of 60,000 and a test set of 10,000 each consisting of 28 by 28 pixels. Each pixel has a value between 0 and 255 (white to black).

(%% Add Math - Dan)
(%% Include description of pre-processing if we do own images - Bram)

### Computation graph
First, the serial implementation is benchmarked on Odyssey for different problem sizes. The code used for this is included in [Code_Serial.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Serial.py). As shown in the Figure below, runtime scales linearly with the number of samples studied. The model reaches above 70% accuracy for the larger training sets. 

(%% Run timing on the serial code - Bram)
(%% Add figure showing different parts of the code and indicate which parts can be parallelized: "try to find the treshold which makes the speedup saturate" - Bram / Data Parallelism)

![Serial-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Sizes_Serial.png)

### OpenMP + MPI parallelization
OpenMP parallelization is implemented using Cython. The inner loops of the learning algorithms are parallelized using nogil pranges that  use static scheduling. By just parallelizing the inner loops using OpenMP, the outer loops can later be parallelized using MPI. The code is compiled using `gcc` (5.2.0) and benchmarked for different numbers of cores. Using Cython with just one thread (so no parallelization) already gives a large speedup compared to the regular serial code. For all problem sizes, speedups around 66x are found compared to the serial code. This speedup is due to ability of Cython to run in C. The Figure below shows the speedup and scaled speedup for 1-10 cores using *n = 1000* as the base case. Using both 5 and 10 cores leads to regular speedup above 1x. Using 10 cores is slower than using 5 cores due to additional overhead (this is only true for *n = 1000*). However, when looking at the scaled speedup (increasing the problem size with the same ratio as increasing the number of processors), using 10 cores has the highest speedup. The brown line shows the efficiencies obtained with the different numbers of cores. Efficiencies are relatively small as a large number of pranges is used, each with their own overhead/allocation costs. As the problem size is relatively small, these overheads are relatively large. In addition to that, part of the module is not parallelized, leaving part of the code to be run in serial in all cases. The used Python script is [Code_OpenMP.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_OpenMP.py) together with Cython module [train_ml_prange.pyx](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/train_ml_prange.pyx).

![OpenMP-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_OpenMP.png)

On top of the inner loop parallelization using OpenMP, MPI parallelization is implemented on the outer loop. This is implemented using the mpi4py package. The current associated Cython module is [train_ml_MPI.pyx](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/train_ml_MPI.pyx).

(%% Add workflow graph /  computation graph -> Analyze performance network, Tcomm/Tcomp, evaluate overhead)
(%% Describe Odyssey architecture used)
(%% 2 Benchmarks [1 MPI = Lambda, 2 MPI = Data] including running on 8 compute nodes)
(%% Evaluate: Amdahl Law + GUstafson Law (Strong + Weak scaling) + throughput/speedup/efficiency/iso-efficiency)

(%% Add benchmark with Intel compiler)
(%% Add benchmark with dynamic instead of static scheduling)

### Spark parallelization
Spark allows a different method of parallelizing the learning algorithm. Using functional parallelism, Spark parallelizes using compositions of functions. A Spark version of the code is implemented on the Amazon Web Services (AWS) EMR Spark cluster. The code is run with 1 master and 2 core nodes and validate that it gives the same results as the serial implementation. The resulting speedup compared to running the serial Odyssey code is shown in the figure below.

(%% Describe AWS architecture used)
(%% Evaluate: Amdahl Law + Gustafson Law (Strong + Weak scaling) + throughput/speedup/efficiency/iso-efficiency)
(%% Add GPU acceleration?)
(%% Add explanation for drop in speedup in Spark -> Will update Spark Benchmark)

![Spark-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_Spark.png)

Speedup larger than 1x is found for all problem sizes studied. More work will be done on applying Spark to the loop over the Tikhonov regularization factors and benchmarking it for varying hardware setups on AWS. The Spark version of the code is [Code_Spark.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Spark.py). Python code used for all the plots is included in the code directory.

### Conclusions
(%% Add conclusions and potentially add suggestions for future work)
