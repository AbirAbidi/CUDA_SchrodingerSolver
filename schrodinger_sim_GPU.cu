#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

#define N 1000000      // number of spatial points
#define NSTEPS 1000    // number of time steps
#define DX 0.000001f   // smaller spacing to cover wider range
#define DT 0.0001f
#define PI 3.14159265358979323846
#define HBAR 1.0f
#define MASS 1.0f

__global__ void evolve_gpu(float* real, float* imag, float* V, float dx, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < N - 1) {
        float r = real[i];
        float im = imag[i];

        float laplacian_r = real[i-1] - 2*r + real[i+1];
        float laplacian_im = imag[i-1] - 2*im + imag[i+1];

        real[i] = r - dt * (-(HBAR*HBAR)/(2*MASS) * laplacian_im / (dx*dx) + V[i]*im);
        imag[i] = im + dt * (-(HBAR*HBAR)/(2*MASS) * laplacian_r / (dx*dx) + V[i]*r);
    }
}

void initialize_wave(float* real, float* imag, float* V) {
    for (int i = 0; i < N; i++) {
        float x = i * DX;

        float packet1_real = expf(-500.0f * (x - 0.3f)*(x - 0.3f)) * cosf(100.0f * x);
        float packet1_imag = expf(-500.0f * (x - 0.3f)*(x - 0.3f)) * sinf(100.0f * x);

        float packet2_real = expf(-500.0f * (x - 0.7f)*(x - 0.7f)) * cosf(-100.0f * x);
        float packet2_imag = expf(-500.0f * (x - 0.7f)*(x - 0.7f)) * sinf(-100.0f * x);

        real[i] = packet1_real + packet2_real;
        imag[i] = packet1_imag + packet2_imag;
        V[i] = 0.0f;

        if (x > 0.5f && x < 0.51f) V[i] = 1000.0f;  // Potential barrier
    }
}

void save_data(float* real, float* imag) {
    std::ofstream file("wave_output_gpu.txt");
    for (int i = 0; i < N; i += 1) {
      // calculates the probability density for each point |psi(x,t)|^2 = real^2 + imag^2
        float prob = real[i]*real[i] + imag[i]*imag[i];
        file << prob << "\n";
    }
    file.close();
}

int main() {
    float *real_cpu = new float[N];
    float *imag_cpu = new float[N];
    float *V_cpu = new float[N];

    initialize_wave(real_cpu, imag_cpu, V_cpu);

    float *real_gpu, *imag_gpu, *V_gpu;
    cudaMalloc(&real_gpu, N * sizeof(float));
    cudaMalloc(&imag_gpu, N * sizeof(float));
    cudaMalloc(&V_gpu, N * sizeof(float));

    cudaMemcpy(real_gpu, real_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(imag_gpu, imag_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_gpu, V_cpu, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NSTEPS; t++) {
        evolve_gpu<<<gridSize, blockSize>>>(real_gpu, imag_gpu, V_gpu, DX, DT);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "GPU time: " << duration.count() << " seconds\n";

    cudaMemcpy(real_cpu, real_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag_cpu, imag_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);

    save_data(real_cpu, imag_cpu);

    delete[] real_cpu;
    delete[] imag_cpu;
    delete[] V_cpu;
    cudaFree(real_gpu);
    cudaFree(imag_gpu);
    cudaFree(V_gpu);

    return 0;
}
