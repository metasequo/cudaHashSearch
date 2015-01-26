# cudaHashSearch
GPGPUを用いて文字列探索をする。  
文字列探索の手法にはラビン・カープ法を用いる。  
ラビン・カープ法とは、文字列のハッシュ値を計算し、パーンを探索する手法である。  

##開発環境
* 言語/ライブラリ : CUDA 6.5
* エディタ/コンパイラ : Visual Studio Express 2013 for Windows Desktop
* CPU : Intel Core i7 3770
* MotherBoard : ASUS P8Z77-V PRO
* RAM : PC3-12800(DDR3-1600) 4GB * 2
* GraphicBoard : MSI N660GTX Twin Frozr III OC
    * GPU : NVIDIA GTX 660
    * VRAM : GDDR5 2048MB
    * CUDA Cores : 960 CUDA Cores (5 Multiprocessors, 192 CUDA Cores/MP)
    * GPU Clock rate : 1098 MHz
    * Memory Clock rate : 3004 MHz
    * L2 Cache Size : 393216 Bytes
    * Warp Size : 32

##プログラムについて
GPUでラビン・カープ法を実装するにあたって、工夫した点は以下の点である。
### 1. ハッシュ値計算を並列演算を利用し、高速に処理する。
CPUで行うラビン・カープ法では、ハッシュ値計算にローリングハッシュ法を用いるが、並列計算には向かない。  
GPUが得意とする単純な計算である、足し算でハッシュ値を計算する。  
ハッシュ計算を行うプログラムは以下の通りである。  
```C++
/* Hash Calculation */
__global__ void gHashCalc(char *text, int *length, unsigned int *rehash)
{
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int scan_idx;

	rehash[col_idx] = 0;

	for (scan_idx = 0; scan_idx < *length; scan_idx++)
	{
		rehash[col_idx] += ((scan_idx + 1) * RADIX) * text[col_idx + scan_idx];
		/* RADIX = 8209 */
		__syncthreads();
	}
	__syncthreads();
}
```
ハッシュ値は、パターン文字列長回ループする中での総和としている。  
何文字目なのかに素数(このプログラムでは8209)を掛け、更に文字コードを掛けることで、ハッシュ値の衝突が少なくなるようにしている。  
