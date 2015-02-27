#include <iostream>
#include <string>

#include <cuda_runtime.h>

using namespace std;

#define	RADIX 257
#define SIZE 9216
#define BLOCK_SIZE 512
#define GRID_SIZE 64
#define LOOP_NUM 100

//int HashCalc(char *text, int length);
__host__ void hHashCalc(char *text, int length, unsigned int *rehash);
__global__ void gHashCalc(char *text, int *length, unsigned int *rehash);
__device__ void dHashCalc(char *text, int *length, unsigned int *rehash);
// void textHash(char *text, int textlen, int texthas [], int patlen);
__global__ void textHash(char *text, int *textlen, unsigned int *texthas, int *patlen);
void HashSearch(char *text, int textlen, unsigned int texthas [], char *pattern, int patlen, unsigned int pathas, bool flag []);
__global__ void gHashSearch(char *text, int *textlen, unsigned int *texthas, char *pattern, int *patlen, unsigned int *pathas, bool *flag);
void Emphasis(char *text, int textlen, int patlen, bool flag [], int Count);
void InsertChar(char *text, char *shift, bool flag [], bool mem [], int *counter, char *insert);
void ShiftChar(char *text, char *shift1, char *shift2, bool flag [], bool mem1 [], bool mem2 [], int *counter, int inslen, int looptimes);

int main(){
	char text[SIZE * 2], pattern[SIZE];
	string inputtext;
	int textlen[1], patlen[1];
	unsigned int texthas[SIZE * 2] = { 0 }, pathas[1] = { 0 };
	bool FoundFlag[SIZE] = { 0 };
	int FoundCount = 0;
	int i;
//	cout << "*Please input text." << endl;
	getline(cin, inputtext);
	const char *convert = inputtext.c_str();
	strcpy(text, convert);
	textlen[0] = strlen(text);
//	cout << textlen[0] << endl;

	do{
//		cout << endl << "*Please input pattern." << endl;
		getline(cin, inputtext);
		convert = inputtext.c_str();
		strcpy(pattern, convert);
		patlen[0] = strlen(pattern);


		if (textlen[0] < patlen[0])
		{
			cout << "**Search pattern is larger than the text size.**" << endl;
		}
	} while (textlen[0] < patlen[0]);

	//GPU用変数
	char *dText, *dPattern;
	unsigned int *dTexthas, *dPathas;
	int *dTextlen, *dPatlen;
	bool *dFoundFlag;

	cudaMalloc((void**) &dText, sizeof(char)*SIZE);
	cudaMemcpy(dText, text, sizeof(char)*SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &dPattern, sizeof(char)*SIZE);
	cudaMemcpy(dPattern, pattern, sizeof(char)*SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &dTexthas, sizeof(unsigned int)*SIZE);
	cudaMemcpy(dTexthas, texthas, sizeof(unsigned int)*SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &dPathas, sizeof(unsigned int)*SIZE);
	cudaMemcpy(dPathas, pathas, sizeof(unsigned int)*SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &dTextlen, sizeof(int)*SIZE);
	cudaMemcpy(dTextlen, textlen, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &dPatlen, sizeof(int)*SIZE);
	cudaMemcpy(dPatlen, patlen, sizeof(int)*SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &dFoundFlag, sizeof(int)*SIZE);
	cudaMemcpy(dFoundFlag, FoundFlag, sizeof(bool)*SIZE, cudaMemcpyHostToDevice);


	for (int cnt = 1; cnt <= GRID_SIZE * BLOCK_SIZE; cnt++){
		float Time_pathas = 0;
		float Time_memcpy1 = 0;
		float Time_texthas = 0;
		float Time_memcpy2 = 0;
		float Time_sum = 0;
		float Time_HashSearch = 0;

		dim3 grid(GRID_SIZE * BLOCK_SIZE / cnt);
		dim3 block(cnt);
		while (GRID_SIZE * BLOCK_SIZE % cnt != 0)
		{
			cnt++;
			dim3 grid(GRID_SIZE * BLOCK_SIZE / cnt);
			dim3 block(cnt);
		}

		//タイマーの設定
		//		cout << "Calculation start in the GPU." << endl;
		//		cout << "BlockSize\t:\t" << GRID_SIZE * BLOCK_SIZE / cnt << "\nGridSize\t:\t" << cnt << endl;
		cout << "BlockSize," << GRID_SIZE * BLOCK_SIZE / cnt << "\nGridSize," << cnt << endl;
		//	float sum = 0.0f;

		for (int loopcnt = 0; loopcnt < LOOP_NUM; loopcnt++){
/*			if (loopcnt == 50){
				dim3 grid(cnt);
			}
			*/
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float millseconds = 0.0f, sum = 0.0f;

			//パターンのハッシュ値計算
			cudaEventRecord(start, 0);

			gHashCalc <<<grid, block>>> (dPattern, dPatlen, dPathas);
			cudaThreadSynchronize();

			//	/*
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&millseconds, start, stop);
			sum += millseconds;
			Time_pathas += millseconds;
			//		cout << "Time required(pattern hash)\t:\t" << millseconds << " millseconds" << endl;

			cudaEventRecord(start, 0);
			//	*/

			cudaMemcpy(pathas, dPathas, sizeof(unsigned int), cudaMemcpyDeviceToHost);

			//	/*
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&millseconds, start, stop);
			sum += millseconds;
			Time_memcpy1 += millseconds;
			//		cout << "Time required(memcpy)\t:\t" << millseconds << " millseconds" << endl;

			//	*/

			//		cout << endl << "*Pattern Hash(" << pattern << ") = " << pathas[0] << endl << endl;

			//テキストのハッシュ値計算
			cudaEventRecord(start, 0);

			textHash <<<grid, block>>> (dText, dTextlen, dTexthas, dPatlen);
			cudaThreadSynchronize();

			//	/*
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&millseconds, start, stop);
			sum += millseconds;
			Time_texthas += millseconds;
			//		cout << "Time required(text hash)\t:\t" << millseconds << " millseconds" << endl;

			cudaEventRecord(start, 0);
			//	*/

			cudaMemcpy(texthas, dTexthas, sizeof(unsigned int)*SIZE, cudaMemcpyDeviceToHost);

			//	/*
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&millseconds, start, stop);
			sum += millseconds;
			Time_memcpy2 += millseconds;
			//		cout << "Time required(memcpy)\t:\t" << millseconds << " millseconds" << endl;
			//	*/

			Time_sum += sum;
			//		cout << "Time required(sum)\t:\t" << sum << " millseconds" << endl;

			//ハッシュ値比較
			//		cout << "*Finding..." << endl;

			cudaEventRecord(start, 0);

			HashSearch(text, textlen[0], texthas, pattern, patlen[0], pathas[0], FoundFlag);
			//gHashSearch << <grid, block >> > (dText, dTextlen, dTexthas, dPattern, dPatlen, dPathas, dFoundFlag);
			//cudaMemcpy(FoundFlag, dFoundFlag, sizeof(bool)*SIZE, cudaMemcpyDeviceToHost);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&millseconds, start, stop);
			sum += millseconds;
			Time_HashSearch += millseconds;
			//		cout << "Time required(HashSearch)\t:\t" << millseconds << " millseconds" << endl;

		}
		cout << "Time required(pattern hash)," << Time_pathas / LOOP_NUM << endl;
		cout << "Time required(memcpy1)," << Time_memcpy1 / LOOP_NUM << endl;
		cout << "Time required(text hash)," << Time_texthas / LOOP_NUM << endl;
		cout << "Time required(memcpy2)," << Time_memcpy2 / LOOP_NUM << endl;
		cout << "Time required(sum)," << Time_sum / LOOP_NUM << endl;
		cout << "Time required(HashSearch)," << Time_HashSearch / LOOP_NUM << endl;



	}
/*
	for (i = 0; i < textlen[0]; i++){
		//cout << "*Text Hash(";
		//for (int j = 0; j < patlen[0]; j++) cout << text[i + j];
		//cout << ") = " << texthas[i] << endl;
		if (FoundFlag[i] == true)	FoundCount++;
	}
	cout << "*FoundCount = " << FoundCount << endl;
	if (FoundCount != 0)
	{
		Emphasis(text, textlen[0], patlen[0], FoundFlag, FoundCount);
		cout << endl << "**Found!!**" << endl << text << endl;
	}
	else
	{
		cout << endl << "**Not found**" << endl;
	}

*/
	
	cudaFree(dText);
	cudaFree(dPattern);
	cudaFree(dTexthas);


	return 0;
}

__host__ void hHashCalc(char *text, int length, unsigned int *rehash)
{
	int scan_idx;
	*rehash = 0;

	for (scan_idx = 0; scan_idx < length; scan_idx++)
	{
		//		rehash = rehash * RADIX + text[i];
		//		rehash += (pow(RADIX, (double)i)) * text[i];
		*rehash += ((scan_idx + 1) * RADIX) * text[scan_idx];
	}

	/*
	cout << "Hash(";
	for(i = 0; i < length; i++) cout << text[i];
	cout << ") = " << rehash << endl;
	*/

}

__global__ void gHashCalc(char *text, int *length, unsigned int *rehash)
{
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int scan_idx;

	rehash[col_idx] = 0;

	for (scan_idx = 0; scan_idx < *length; scan_idx++)
	{
		rehash[col_idx] += ((scan_idx + 1) * RADIX) * text[col_idx + scan_idx];
		//		*rehash += ((scan_idx + 1) * RADIX) * text[scan_idx];
		//__syncthreads();
	}
	//__syncthreads();
}

__device__ void dHashCalc(char *text, int *length, unsigned int *rehash)
{
	//	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int scan_idx;

	*rehash = 0;

	for (scan_idx = 0; scan_idx < *length; scan_idx++)
	{
		//		*rehash += ((scan_idx + 1) * RADIX) * text[col_idx + scan_idx];
		*rehash += ((scan_idx + 1) * RADIX) * text[scan_idx];
		__syncthreads();
	}

	/*
	cout << "Hash(";
	for(i = 0; i < length; i++) cout << text[i];
	cout << ") = " << rehash << endl;
	*/
}

__global__ void textHash(char *text, int *textlen, unsigned int *texthas, int *patlen)
{
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int scan_idx;

	texthas[col_idx] = 0;
	for (scan_idx = 0; scan_idx < *patlen; scan_idx++){
		texthas[col_idx] += ((scan_idx + 1) * RADIX) * text[col_idx + scan_idx];
		//__syncthreads();
	}
	//__syncthreads();
}


__global__ void gHashSearch(char *text, int *textlen, unsigned int *texthas, char *pattern, int *patlen, unsigned int *pathas, bool *flag)
{
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (*pathas == texthas[col_idx])
	{
	int scan_idx = 0;
		do{
			if (text[col_idx + scan_idx] != pattern[scan_idx])	break;
		} while (++scan_idx < *patlen);

		if (scan_idx == *patlen)
		{
			flag[col_idx] = true;
		}
	}
}


void HashSearch(char *text, int textlen, unsigned int texthas [], char *pattern, int patlen, unsigned int pathas, bool flag [])
{
	int i, j;

	for (i = 0; i < textlen - patlen + 1; i++)
	{
		if (pathas == texthas[i])
		{
			//cout << "Found the same hash!" << endl;
	/*		cout << "Text Hash(";
			for (j = 0; j < patlen; j++) cout << text[i + j];
			cout << ") = " << texthas[i] << endl;
	*/
			j = 0;
			do{
				if (text[i + j] != pattern[j])	break;
			} while (++j < patlen);

			if (j == patlen)
			{
				flag[i] = true;
			}
		}
	}
}

void Emphasis(char *text, int textlen, int patlen, bool flag [], int Count)
{
	int i, looptimes;
	char shift1[SIZE], shift2[SIZE];
	bool mem1[SIZE * 2], mem2[SIZE * 2];
	char insert1 [] = " << ", insert2 [] = " >> ";
	int inslen = strlen(insert1);

	looptimes = textlen - patlen + (8 * Count);

	for (i = 0; i < textlen - patlen + (8 * Count); i++)
	{
		if (flag[i] == true)
		{
			InsertChar(text, shift1, flag, mem1, &i, insert1);
			ShiftChar(text, shift1, shift2, flag, mem1, mem2, &i, inslen, looptimes);
			i += patlen;

			InsertChar(text, shift1, flag, mem1, &i, insert2);
			ShiftChar(text, shift1, shift2, flag, mem1, mem2, &i, inslen, looptimes);

		}
	}
}

void InsertChar(char *text, char *shift, bool flag [], bool mem [], int *counter, char *insert)
{
	int inslen = strlen(insert), j;
	for (j = 0; j < inslen; j++)
	{
		shift[j] = text[*counter + j];
		mem[j] = flag[*counter + j];
	}
	for (j = 0; j < inslen; j++)
	{
		text[*counter + j] = insert[j];
	}
	*counter += inslen;
}

void ShiftChar(char *text, char *shift1, char *shift2, bool flag [], bool mem1 [], bool mem2 [], int *counter, int inslen, int looptimes)
{
	int j;
	for (j = 0; j < looptimes; j++)
	{
		shift2[j] = text[*counter + j];
		mem2[j] = flag[*counter + j];
		if (j < inslen){
			text[*counter + j] = shift1[j];
			flag[*counter + j] = mem1[j];
		}
		else{
			text[*counter + j] = shift2[j - inslen];
			flag[*counter + j] = mem2[j - inslen];
		}
	}
}

