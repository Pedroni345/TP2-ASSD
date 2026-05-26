#include <complex.h>   // float complex, cexpf, crealf, cimagf
#include <stdlib.h>    // malloc, free
#include <stddef.h>    // size_t
#include <stdio.h>     // printf
#include <math.h>      // fabsf, log2f

#define PI 3.14159265358979323846f

/* =========================================================
 *  DFT de referencia O(n^2)  — provista por la cátedra
 * ========================================================= */
void dft(float complex *in, float complex *out, size_t N) {
    float complex *output = (float complex *)malloc(sizeof(float complex) * N);
    for (size_t k = 0; k < N; k++) {
        output[k] = 0;
        for (size_t n = 0; n < N; n++) {
            output[k] += in[n] * cexpf(-2 * PI * I * k * n / N);
        }
    }
    for (size_t k = 0; k < N; k++) {
        out[k] = output[k];
    }
    free(output);
}

/* =========================================================
 *  FFT Cooley-Tukey — decimación en tiempo (DIT), in-place
 *
 *  @param in   Puntero al buffer de entrada.
 *  @param out  Puntero al buffer de salida (puede ser == in).
 *  @param N    Cantidad de puntos (potencia de 2, 1 <= N <= 4096).
 * ========================================================= */
void fft(float complex *in, float complex *out, size_t N) {
	
	/* --------- Caso N malo --------------*/
	if (N == 0 || (N & (N - 1)) != 0) {
        printf("Error: N (%zu) no es una potencia de 2.\n", N);
        
        for (size_t i = 0; i < N; i++) {
            out[i] = 0.0f + 0.0f * I;
        }
        return;
    }

    /* --- Caso base ---------------- */
    if (N == 1) {
        out[0] = in[0];
        return;
    }

    /* --- Copiar al buffer de salida si son distintos ------- */
    if (in != out) {
        for (size_t i = 0; i < N; i++) {
            out[i] = in[i];
        }
    }

    /* --- Bit-reversal permutation -------------------------- */
    size_t bits = (size_t)(log2f((float)N));   /* log2 de N  */
    
    for (size_t i = 0; i < N; i++) {
    	
        size_t j = 0;
        size_t tmp = i;
        for (size_t b = 0; b < bits; b++) {
            j = (j << 1) | (tmp & 1);
            tmp >>= 1;
        }
        
        if (j > i) {
            float complex swap = out[i];
            out[i] = out[j];
            out[j] = swap;
        }
    }
    
    printf("\n--- Después de Bit-Reversal Permutation ---\n");
	for (size_t i = 0; i < N; i++) {
    	printf("  out[%2zu] = %.4f %+.4f j\n", i, crealf(out[i]), cimagf(out[i]));
	}

    /* --- Butterfly iterativo (bottom-up) ------------------- */
    for (size_t len = 2; len <= N; len <<= 1) {
    float complex W = cexpf(-2.0f * PI * I / (float)len);

    // AGREGAR ESTO:
    printf("\n=== Etapa len = %zu ===\n", len);

    for (size_t k = 0; k < N; k += len) {
        float complex Wn = 1.0f + 0.0f * I;

        for (size_t n = 0; n < len / 2; n++) {
            float complex u = out[k + n];
            float complex v = out[k + n + len / 2] * Wn;

            // AGREGAR ESTO (antes de modificar out[]):
            printf("  Butterfly [%2zu, %2zu]: u=(%.4f%+.4fj)  v=(%.4f%+.4fj)  => "
                   "superior=(%.4f%+.4fj)  inferior=(%.4f%+.4fj)\n",
                   k + n, k + n + len / 2,
                   crealf(u), cimagf(u),
                   crealf(v), cimagf(v),
                   crealf(u + v), cimagf(u + v),
                   crealf(u - v), cimagf(u - v));

            out[k + n] = u + v;
            out[k + n + len / 2] = u - v;
            Wn *= W;
        }
    }

    // AGREGAR ESTO (estado del vector al final de cada etapa):
    printf("  Estado del vector al final de etapa len=%zu:\n", len);
    for (size_t i = 0; i < N; i++) {
        printf("    out[%2zu] = %.4f %+.4f j\n", i, crealf(out[i]), cimagf(out[i]));
    }
}
    
    
}

/* =========================================================
 *  Utilidades de impresión
 * ========================================================= */

static void print_input_vector(const char *label, float complex *v, size_t N) {
    printf("%s (N = %zu):\n", label, N);
    for (size_t n = 0; n < N; n++) {
        float re = crealf(v[n]);
        float im = cimagf(v[n]);
        if (fabsf(re) < 1e-4f) re = 0.0f;
        if (fabsf(im) < 1e-4f) im = 0.0f;
        printf("  x[%2zu]  =  %8.4f  %+8.4f j\n", n, re, im);
    }
    printf("\n");
}


/* Imprime un vector complejo con formato legible */
static void print_complex_vector(const char *label,
                                  float complex *v, size_t N) {
    printf("%s:\n", label);
    for (size_t k = 0; k < N; k++) {
        float re = crealf(v[k]);
        float im = cimagf(v[k]);
        /* Suprime el signo negativo en cero para mayor claridad */
        if (fabsf(re) < 1e-4f) re = 0.0f;
        if (fabsf(im) < 1e-4f) im = 0.0f;
        printf("  [%2zu]  %8.4f  %+8.4f j\n", k, re, im);
    }
}

/* Calcula el error máximo entre dos vectores */
static float max_error(float complex *a, float complex *b, size_t N) {
    float max_err = 0.0f;
    for (size_t k = 0; k < N; k++) {
        float err = cabsf(a[k] - b[k]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* =========================================================
 *  Prueba 1 — señal de impulso: X[k] = 1 para todo k
 * ========================================================= */
static void test_impulso(void) {
    const size_t N = 8;
    float complex in[8], out_dft[8], out_fft[8];

    printf("========================================\n");
    printf(" TEST 1: Impulso unitario  (N = %zu)\n", N);
    printf("========================================\n");

    for (size_t n = 0; n < N; n++) in[n] = (n == 0) ? 1.0f : 0.0f; //Inicializa el pulso unitario agregando un 1 real en el primer elemento

	
    dft(in, out_dft, N);
    fft(in, out_fft, N);

    print_complex_vector("DFT", out_dft, N);
    print_complex_vector("FFT", out_fft, N);
    printf("Error maximo: %.2e\n\n", max_error(out_dft, out_fft, N));
}

/* =========================================================
 *  Prueba 2 — senoidal pura: x[n] = sin(2pi * k0 * n / N)
 *  Se espera que solo dos bins tengan energia.
 * ========================================================= */
static void test_senoidal(void) {
    const size_t N  = 16;
    const size_t k0 = 3;          /* frecuencia del seno (bin) */
    float complex in[16], out_dft[16], out_fft[16];

    printf("========================================\n");
    printf(" TEST 2: Senoidal pura  (N=%zu, k0=%zu)\n", N, k0);
    printf("========================================\n");

    for (size_t n = 0; n < N; n++) {
        in[n] = sinf(2.0f * PI * k0 * n / N) + 0.0f * I;
    }
    
    print_input_vector("Señal de entrada", in, N);

    dft(in, out_dft, N);
    fft(in, out_fft, N);

    print_complex_vector("DFT", out_dft, N);
    print_complex_vector("FFT", out_fft, N);
    printf("Error maximo: %.2e\n\n", max_error(out_dft, out_fft, N));
}

/* =========================================================
 *  Prueba 3 — buffer in-place (in == out)
 * ========================================================= */
static void test_inplace(void) {
    const size_t N = 8;
    float complex buf_fft[8], out_dft[8], ref[8];

    printf("========================================\n");
    printf(" TEST 3: FFT in-place  (N = %zu)\n", N);
    printf("========================================\n");

    /* Señal: rampa */
    for (size_t n = 0; n < N; n++) {
        ref[n] = buf_fft[n] = (float)n + 0.0f * I;
    }

    dft(ref, out_dft, N);
    fft(buf_fft, buf_fft, N);   /* in == out */

    print_complex_vector("DFT",      out_dft, N);
    print_complex_vector("FFT in-place", buf_fft, N);
    printf("Error maximo: %.2e\n\n", max_error(out_dft, buf_fft, N));
}

/* =========================================================
 *  Prueba 4 — N = 1 (caso base)
 * ========================================================= */
static void test_n1(void) {
    float complex in[1]     = { 3.14f + 2.71f * I };
    float complex out_dft[1], out_fft[1];

    printf("========================================\n");
    printf(" TEST 4: N = 1  (caso base)\n");
    printf("========================================\n");

    dft(in, out_dft, 1);
    fft(in, out_fft, 1);

    printf("Entrada:  %.4f %+.4f j\n", crealf(in[0]), cimagf(in[0]));
    printf("DFT:      %.4f %+.4f j\n", crealf(out_dft[0]), cimagf(out_dft[0]));
    printf("FFT:      %.4f %+.4f j\n", crealf(out_fft[0]), cimagf(out_fft[0]));
    printf("Error maximo: %.2e\n\n", max_error(out_dft, out_fft, 1));
}

/* =========================================================
 *  main
 * ========================================================= */
int main(void) {
    printf("\n");
    printf("  Comparacion DFT (referencia O(n^2)) vs FFT (Cooley-Tukey)\n");
    printf("  Materia: 25.20 - Analisis de Senales y Sistemas Digitales\n\n");

    test_impulso();
    test_senoidal();
    test_inplace();
    test_n1();

    printf("Todos los tests completados.\n");
    return 0;
}
