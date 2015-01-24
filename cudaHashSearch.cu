#include <iostream>
#include <string>

#include <cuda_runtime.h>

using namespace std;

#define	RADIX 257
#define SIZE 4096
#define BLOCK_SIZE 128
#define GRID_SIZE 16

//int HashCalc(char *text, int length);
__host__ void hHashCalc(char *text, int length, unsigned int *rehash);
__global__ void gHashCalc(char *text, int *length, unsigned int *rehash);
__device__ void dHashCalc(char *text, int *length, unsigned int *rehash);
// void textHash(char *text, int textlen, int texthas [], int patlen);
__global__ void textHash(char *text, int *textlen, unsigned int *texthas, int *patlen);
void HashSearch(char *text, int textlen, unsigned int texthas [], char *pattern, int patlen, unsigned int pathas, int flag []);
void Emphasis(char *text, int textlen, int patlen, int flag [], int Count);
void InsertChar(char *text, char *shift, int flag [], int mem [], int *counter, char *insert);
void ShiftChar(char *text, char *shift1, char *shift2, int flag [], int mem1 [], int mem2 [], int *counter, int inslen, int looptimes);

int main(){
	char text[SIZE], pattern[SIZE];
	string inputtext;
	int textlen[1], patlen[1];
	unsigned int texthas[SIZE * 2] = { 0 }, pathas[1] = { 0 };
	int FoundFlag[SIZE] = { 0 }, FoundCount = 0;
	int i;
	cout << "*Please input text." << endl;
	getline(cin, inputtext);
	cout << endl << "*Please input pattern." << endl;
	cin >> pattern;

	const char *convert = inputtext.c_str();
	strcpy(text, convert);

	textlen[0] = strlen(text);
	patlen[0] = strlen(pattern);

	/*
	for(i = 0; i < patlen; i++)
	{
	FoundFlag[i] = HashCalc(&pattern[i], 1);
	cout << "Hash(" << patterin[i] << ") = " << FoundFlag[i] << endl;
	}
	*/

	if (textlen < patlen)
	{
		cout << "**Search pattern is larger than the text size.**" << endl;
	}
	else
	{

		//GPU
		char *dText, *dPattern;
		unsigned int *dTexthas, *dPathas;
		int *dTextlen, *dPatlen;
		//		int *dFoundFlag;

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

		//		cudaMalloc((void**) &dFoundFlag, textlen);
		//		cudaMemcpy(dFoundFlag, FoundFlag, textlen, cudaMemcpyHostToDevice);

		dim3 grid(GRID_SIZE);
		dim3 block(BLOCK_SIZE);

		gHashCalc << <grid, block >> >(dPattern, dPatlen, dPathas);
		cudaThreadSynchronize();

		cudaMemcpy(pathas, dPathas, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cout << endl << "*Pattern Hash(" << pattern << ") = " << pathas[0] << endl << endl;

		//hHashCalc(pattern, patlen[0], pathas);
		//cout << endl << "*Pattern Hash(" << pattern << ") = " << pathas[0] << endl << endl;

		cout << "*Finding..." << endl;


		textHash << <grid, block >> >(dText, dTextlen, dTexthas, dPatlen);
		cudaThreadSynchronize();

		cudaMemcpy(texthas, dTexthas, sizeof(unsigned int)*SIZE, cudaMemcpyDeviceToHost);


		HashSearch(text, textlen[0], texthas, pattern, patlen[0], pathas[0], FoundFlag);
		for (i = 0; i < textlen[0]; i++){
			//cout << "*Text Hash(";
			//for (int j = 0; j < patlen[0]; j++) cout << text[i + j];
			//cout << ") = " << texthas[i] << endl;
			if (FoundFlag[i] == 1)	FoundCount++;

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

		cudaFree(dText);
		cudaFree(dPattern);
		cudaFree(dTexthas);

	}



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
		__syncthreads();
	}
	__syncthreads();
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
	unsigned int scan_idx;


	for (scan_idx = 0; scan_idx < *textlen /*- *patlen + 1*/; scan_idx++){
		//		dHashCalc(&text[scan_idx], patlen, &texthas[scan_idx]);
		texthas[col_idx] = 0;
		for (int i = 0; i < *patlen; i++){
			texthas[col_idx] += ((i + 1) * RADIX) * text[col_idx + i];
			__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
}

void HashSearch(char *text, int textlen, unsigned int texthas [], char *pattern, int patlen, unsigned int pathas, int flag [])
{
	int i, j;

	for (i = 0; i < textlen - patlen + 1; i++)
	{
		if (pathas == texthas[i])
		{
			cout << "Found the same hash!" << endl;
			cout << "Text Hash(";
			for (j = 0; j < patlen; j++) cout << text[i + j];
			cout << ") = " << texthas[i] << endl;
			j = 0;
			do{
				if (text[i + j] != pattern[j])	break;
			} while (++j < patlen);

			if (j == patlen)
			{
				flag[i] = 1;
			}
		}
	}
}

void Emphasis(char *text, int textlen, int patlen, int flag [], int Count)
{
	int i, looptimes;
	char shift1[SIZE], shift2[SIZE];
	int mem1[SIZE * 2], mem2[SIZE * 2];
	char insert1 [] = " << ", insert2 [] = " >> ";
	int inslen = strlen(insert1);

	looptimes = textlen - patlen + (8 * Count);

	for (i = 0; i < textlen - patlen + (8 * Count); i++)
	{
		if (flag[i] == 1)
		{
			InsertChar(text, shift1, flag, mem1, &i, insert1);
			ShiftChar(text, shift1, shift2, flag, mem1, mem2, &i, inslen, looptimes);
			i += patlen;

			InsertChar(text, shift1, flag, mem1, &i, insert2);
			ShiftChar(text, shift1, shift2, flag, mem1, mem2, &i, inslen, looptimes);

		}
	}
}

void InsertChar(char *text, char *shift, int flag [], int mem [], int *counter, char *insert)
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

void ShiftChar(char *text, char *shift1, char *shift2, int flag [], int mem1 [], int mem2 [], int *counter, int inslen, int looptimes)
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

