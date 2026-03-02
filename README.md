# 🚀 GPU-Accelerated Simulation of the Time-Dependent Schrödinger Equation

## 📌 Project Overview

This project implements a numerical simulation of the **time-dependent Schrödinger equation (TDSE)** using both:

* A **CPU implementation (C++)**
* A **GPU implementation (CUDA C++)**

The goal is to compare computational performance while simulating quantum wave packet dynamics in one spatial dimension.

The simulation models two Gaussian wave packets traveling in opposite directions and interacting with a central potential barrier.

---

## 🧠 Physical Background

The time-dependent Schrödinger equation in 1D is:

iħ ∂ψ/∂t = - (ħ² / 2m) ∂²ψ/∂x² + V(x)ψ

Where:

- ψ(x,t) → complex wave function
- |ψ(x,t)|² → probability density
- V(x) → potential energy
- ħ → reduced Planck constant
- m → particle mass

### 🔎 What Does the Schrödinger Equation Predict?

The equation does **not** give the exact position of a particle.

Instead, it computes a **complex wave function**:

ψ(x,t) = Re(ψ) + i Im(ψ)

The observable quantity is:

|ψ(x,t)|² = Re(ψ)² + Im(ψ)²

This represents the probability density of finding the particle at position x at time t.

---

## 🔬 Simulation Description

### 1️⃣ Spatial Discretization

Because continuous space cannot be computed numerically, the spatial domain is discretized:

* **N = 1,000,000 spatial points**
* Uniform spacing **DX**
* Finite difference approximation of the Laplacian

### 2️⃣ Initial Condition

The initial wave function is the sum of two Gaussian wave packets:

* Packet 1 centered at **x = 0.3**
* Packet 2 centered at **x = 0.7**
* Opposite momenta (opposite propagation directions)

Mathematically:

ψ(x,0) =
exp(-500(x-0.3)²) e^{i100x}
+
exp(-500(x-0.7)²) e^{-i100x}

### 3️⃣ Potential Barrier

A potential barrier is placed at:

0.5 < x < 0.51

With:

V(x) = 1000 inside the barrier  
V(x) = 0 elsewhere

This allows observation of:

* Reflection
* Transmission
* Quantum interference

---

## ⚙️ Numerical Method

* Explicit finite-difference time integration
* Second-order spatial derivative approximation
* Real and imaginary parts evolved separately
* Dirichlet-type implicit boundary behavior

---

## 🖥 CPU Implementation

* Standard C++
* Nested time and space loops
* Temporary buffers for stable updating
* Compiled with:

```bash
g++ -O2 schrodinger_sim_CPU.cpp -o sim_cpu
```

### ⏱ Performance

```
CPU time ≈ 9.4 seconds
```

---

## ⚡ GPU Implementation (CUDA)

* Parallel spatial computation
* One thread per spatial grid point
* Massive parallel acceleration
* Compiled with:

```bash
nvcc schrodinger_sim_GPU.cu -o schrodinger_sim_GPU
```

### ⏱ Performance

```
GPU time ≈ 0.046 seconds
```

Speedup ≈ 200×

---

## 📊 Visualization

Probability density is saved to:

* `wave_output_cpu.txt`
* `wave_output_gpu.txt`

Python visualization example:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("wave_output_gpu.txt")
plt.plot(data)
plt.title("|ψ(x)|² after 1000 steps (GPU)")
plt.show()
```

---

## 🧮 Why GPU is Faster

The Schrödinger equation update is **embarrassingly parallel**:

Each spatial point depends only on its neighbors.

GPU advantages:

* Thousands of parallel threads
* High memory bandwidth
* Optimized floating-point throughput

---

## 📁 Project Structure

```
├── schrodinger_sim_GPU.cu
├── schrodinger_sim_CPU.cpp
├── wave_output_gpu.txt
├── wave_output_cpu.txt
└── README.md
```

---

## 📌 Technical Specifications

| Parameter         | Value          |
| ----------------- | -------------- |
| Grid Points (N)   | 1,000,000      |
| Time Steps        | 1000           |
| Spatial Step (DX) | 1e-6 (GPU)     |
| Time Step (DT)    | 1e-4 (GPU)     |
| Language          | C++ / CUDA     |
| Precision         | Single (float) |

---

## 🔍 Scientific Remarks

* The method used is explicit and may suffer from stability constraints.
* A more stable approach would involve:

  * Crank–Nicolson scheme
  * Split-operator Fourier method
  * Implicit integration
* The simulation does not explicitly normalize the wave function.
* Boundary conditions are not strictly enforced.

---

## 🎯 Learning Objectives

This project demonstrates:

* Numerical solution of PDEs
* Finite difference methods
* CUDA parallelization
* Performance benchmarking
* Quantum wave packet dynamics
* GPU vs CPU acceleration comparison

---

## 🏁 Conclusion

This project illustrates how GPU computing dramatically accelerates scientific simulations.

It also provides a hands-on implementation of quantum mechanics principles through high-performance computing.

The Schrödinger equation is not only fundamental in physics — it is also an excellent benchmark for parallel numerical methods.

---


