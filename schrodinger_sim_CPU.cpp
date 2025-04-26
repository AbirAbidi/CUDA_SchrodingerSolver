#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdlib> // for std::isnan

#define N 1000000        // Number of points
#define NSTEPS 1000      // Number of steps
#define DX 0.0000001f     // Grid spacing
#define DT 0.000000000000001f      // Time step size
#define HBAR 1.0f
#define MASS 1.0f

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

void evolve_cpu(float* real, float* imag, float* V, float dx, float dt) {
    float* new_real = new float[N];
    float* new_imag = new float[N];

    for (int t = 0; t < NSTEPS; t++) {
        for (int i = 1; i < N - 1; i++) {  // Avoid boundaries
            float r = real[i];
            float im = imag[i];

            float laplacian_r = real[i-1] - 2*r + real[i+1];
            float laplacian_im = imag[i-1] - 2*im + imag[i+1];

            new_real[i] = r - dt * (-(HBAR*HBAR)/(2*MASS) * laplacian_im / (dx*dx) + V[i]*im);
            new_imag[i] = im + dt * (-(HBAR*HBAR)/(2*MASS) * laplacian_r / (dx*dx) + V[i]*r);
        }


        // Swap
        std::swap(real, new_real);
        std::swap(imag, new_imag);
    }

    delete[] new_real;
    delete[] new_imag;
}

void save_data(float* real, float* imag) {
    std::ofstream file("wave_output_cpu.txt");
    for (int i = 0; i < N; i++) {
        float prob = real[i]*real[i] + imag[i]*imag[i];
        file << prob << "\n";
    }
    file.close();
}

int main() {
    float* real = new float[N];
    float* imag = new float[N];
    float* V = new float[N];

    initialize_wave(real, imag, V);

    auto start = std::chrono::high_resolution_clock::now();

    evolve_cpu(real, imag, V, DX, DT);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "CPU time: " << duration.count() << " seconds\n";

    save_data(real, imag);

    delete[] real;
    delete[] imag;
    delete[] V;

    return 0;
}

