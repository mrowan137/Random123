/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef UTIL_CUDA_H__
#define UTIL_CUDA_H__

#include "util.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// utility macros to check return codes and complain/exit on failure
#define CHECKLAST(MSG) 	do { hipError_t e = hipGetLastError(); if (e != hipSuccess) {fprintf(stderr, "%s:%d: CUDA Error: %s: %s\n", __FILE__, __LINE__, (MSG), hipGetErrorString(e)); exit(1); }} while(0)
#define CHECKCALL(RET)	do { hipError_t e = (RET); if (e != hipSuccess) { fprintf(stderr, "%s:%d: CUDA Error: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while(0)

typedef struct cuda_info {
    int devnum, cores, blocks_per_grid, threads_per_block;
    double cycles;
    struct hipDeviceProp_t dev;
} CUDAInfo;

// If devstr is none, chooses device with most cores.
static CUDAInfo *cuda_init(const char *devstr)
{
    CUDAInfo *tp;
    int i, ndev, cores, devcores;
    double cycles;
    CHECKNOTZERO(tp = (CUDAInfo *) malloc(sizeof(CUDAInfo)));
    CHECKCALL( hipGetDeviceCount(&ndev) );
    devcores = 0;
    if (devstr == NULL)
	devstr = getenv("R123EXAMPLE_ENVCONF_CUDA_DEVICE");
    for (i = 0; i < ndev; i++) {
	struct hipDeviceProp_t cu;
	CHECKCALL( hipGetDeviceProperties (&cu, i) );
	// Number of cores is not available from a query, have to hardwire
	// some knowledge here, from web articles about the various generations
	// SM or SMX, might also find this info in
	// CUDA SDK $CUDA_SAMPLES_DIR/common/inc/helper_cuda.h
	// or https://github.com/NVIDIA/nvidia-docker/blob/master/tools/src/cuda/cuda.go
	cores = cu.multiProcessorCount;

#if defined(CUDART_VERSION) || defined(__HIP_PLATFORM_NVIDIA__)
    // Nvidia hardware
	if (cu.major == 1 && cu.minor >= 0 && cu.minor <= 3) {
	    // 1.0 (G80, G92, aka GTX880, Tesla [CSD]870) to 1.3 (GT200, aka GTX280, Tesla [CS]10xx) have 8 cores per MP
	    cores *= 8;
	} else if (cu.major == 2 && cu.minor == 0) {
	    // 2.0 (G100, aka GTX480, Tesla/Fermi [CSM]20[567]0, and GF110, aka GTX580, M2090)
	    cores *= 32;
	} else if (cu.major == 2 && cu.minor == 1) {
	    // 2.1 (GF104, GF114, GF116 aka GTX [45][56]0)
	    cores *= 48;
	} else if (cu.major == 3) {
	    // 3.0 (Kepler GK104 aka GTX 680), 3.2 (TK1), 3.5 (GK11x, GK20x), 3.7 (GK21x)
	    cores *= 192;
	} else if (cu.major == 5) {
	    // 5.0 (Maxwell GM10x), 5.2 (GM20x), 5.3 (TX1)
	    cores *= 128;
	} else if (cu.major == 6 && cu.minor == 0) {
	    // 6.0 (Pascal P100)
	    cores *= 64;
	} else if (cu.major == 6) {
	    // 6.1 (Pascal 10xx, Titan Xp, P40), 6.2 (Drive PX2 and Tegra)
	    cores *= 128;
	} else if (cu.major == 7) {
	    // 7.[025] (Volta and Turing RTX 20[678]0, Titan RTX, Quadro RTX)
	    cores *= 128;
	}
#elif defined(__HIP_PLATFORM_AMD__)
    // AMD hardware
    if (cu.major == 9) {
        // 9.0 (gfx908 aka MI100)
        cores *= 128;
    }
#endif
    else {
	    int coremultguess = 384;
	    cores *= coremultguess;
	    fprintf(stderr, "WARNING: Unknown number of cores per MP for this device: assuming %d, so cpb calculation will be wrong and choice of blocks/grid might be suboptimal\n", coremultguess);
	}
	/* clockrate is in KHz */
	cycles = 1e3 * cu.clockRate * cores;
	printf("  %d: maj %d min %d %s%s ( %d units @ %g MHz ECC=%d %d cores %g Gcycles/s)\n",
	   i, cu.major, cu.minor, nameclean(cu.name), cu.integrated ? " integrated" : "",
	   cu.multiProcessorCount, cu.clockRate*1e-3, cu.ECCEnabled, cores, cycles*1e-9);
	if (devstr && strstr(cu.name, devstr) == NULL) {
	    dprintf(("skipping device %s\n", cu.name));
	    continue;
	}
	if (cores > devcores) {
	    devcores = cores;
	    tp->devnum = i;
	    tp->cores = cores;
	    tp->cycles = cycles;
	    tp->dev = cu;
	}
    }
    if (devcores == 0) {
	fprintf(stderr, "could not find specified device\n");
	exit(1);
    }
    tp->blocks_per_grid = tp->cores; /* seems like a good guess */
    tp->threads_per_block = tp->dev.warpSize * 2;
    printf("Using GPU device %d, %d cores, %g cycles, will try %d blocks/grid %d threads/block\n",
	   tp->devnum, tp->cores, tp->cycles, tp->blocks_per_grid, tp->threads_per_block);
    CHECKCALL(hipSetDevice(tp->devnum));
    dprintf(("cuda_init done\n"));
    return tp;
}

static void cuda_done(CUDAInfo *tp)
{
    dprintf(("cuda_done\n"));
    free(tp);
}

#endif /* UTIL_CUDA_H__ */
