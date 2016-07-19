#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include "rand_func.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <ammintrin.h>
#include <x86intrin.h>


void fill_up(int level, uint32_t* branches, int32_t key, int32_t* filled, int32_t** index, int32_t* gone_up) {
    // if(level < 0) {
    //     printf("This probably shouldn't have happened. Check boundary!!\n");
    // }
    if((filled[level]% (branches[level]-1)==0) && !gone_up[level] && filled[level] != 0) {
        fill_up(level-1, branches, key, filled, index, gone_up);
        gone_up[level] = 1;
    } else {
        index[level][filled[level]] = key;
        filled[level]++;
        gone_up[level] = 0;
    }
}

void print_tree(int32_t** index, uint32_t* max_nodes, uint32_t* branches, int num_levels) {
    int i, j;
    for (i = 0 ; i < num_levels ; i++) {
        for(j = 0; j < max_nodes[i] * (branches[i] - 1); j++) {
            printf("%d, ",index[i][j]);
        }
        printf("\n");
    }
}

int bin_Search(int32_t* arr, int low, int high, int32_t target) {
    // printf("Bin terms - %d, %d\n", low, high);
    while(low<=high) {
        int mid = (low+high)/2;
        if(arr[mid]==target && arr[mid] != INT_MAX)
            return mid;
        else if (arr[mid]>=target) {
            high = mid-1;
        } else {
            low = mid+1;
        }
    }
    return low;
}

int SSE_Search(int32_t* arr, int res_prev, int fanout, int32_t probe) {
	int res;
	__m128i key = _mm_cvtsi32_si128(probe);
	key = _mm_shuffle_epi32(key, _MM_SHUFFLE(0,0,0,0));

	if(fanout == 5) {
		__m128i lvl_1 = _mm_load_si128(( __m128i *) &arr[res_prev << 2]);
		__m128 cmp_1 = _mm_castsi128_ps(_mm_cmpgt_epi32(key, lvl_1));
		res = _mm_movemask_ps(cmp_1);   // ps: epi32
		res = _bit_scan_forward(res ^ 0x1FF);
		res += (res_prev << 2) + res_prev;
	} else if(fanout == 9) {
		__m128i lvl_2_A = _mm_load_si128(( __m128i *) &arr[res_prev << 3]);
		__m128i lvl_2_B = _mm_load_si128(( __m128i *) &arr[(res_prev << 3) + 4]);
		__m128i cmp_2_A = _mm_cmpgt_epi32(key, lvl_2_A);
		__m128i cmp_2_B = _mm_cmpgt_epi32(key, lvl_2_B);
		__m128i cmp_2 = _mm_packs_epi32(cmp_2_A, cmp_2_B);
		cmp_2 = _mm_packs_epi16(cmp_2, _mm_setzero_si128());
		res = _mm_movemask_epi8(cmp_2);
		res = _bit_scan_forward(res ^ 0x1FFFF);
		res += (res_prev << 3) + res_prev;
	} else if(fanout == 17) {
		// key = _mm_loadl_epi32(input_keys++);   // asm: movd
		// key = _mm_shuffle_epi32(key, 0);
		// compare with 16 delimiters stored in 4 registers
		__m128i del_ABCD = _mm_load_si128(( __m128i *) &arr[res_prev << 4]);
        __m128i del_EFGH = _mm_load_si128(( __m128i *) &arr[(res_prev << 4) + 4]);
        __m128i del_IJKL = _mm_load_si128(( __m128i *) &arr[(res_prev << 4) + 8]);
        __m128i del_MNOP = _mm_load_si128(( __m128i *) &arr[(res_prev << 4) + 12]);
		__m128i cmp_ABCD = _mm_cmpgt_epi32(key, del_ABCD);
		__m128i cmp_EFGH = _mm_cmpgt_epi32(key, del_EFGH);
		__m128i cmp_IJKL = _mm_cmpgt_epi32(key, del_IJKL);
		__m128i cmp_MNOP = _mm_cmpgt_epi32(key, del_MNOP);
		// pack results to 16-bytes in a single SIMD register
		__m128i cmp_A_to_H = _mm_packs_epi32(cmp_ABCD, cmp_EFGH);
		__m128i cmp_I_to_P = _mm_packs_epi32(cmp_IJKL, cmp_MNOP);
		__m128i cmp_A_to_P = _mm_packs_epi16(cmp_A_to_H, cmp_I_to_P);
		// extract the mask the least significant bit
		res = _mm_movemask_epi8(cmp_A_to_P);
		res = _bit_scan_forward(res ^ 0x1FFFFFFFF); 
        res += (res_prev << 4) + res_prev;
	} else {
		printf("Error:Supported fanouts are 5, 9 and 17!\n");
		exit(0);
	}
	return res;
}

void Search_9_5_9(int32_t** index, int32_t* probes, int num_probes, int* outputs) {
	register __m128i root1 = _mm_load_si128(( __m128i *) &index[0][0]);
	register __m128i root2 = _mm_load_si128(( __m128i *) &index[0][4]);
	register __m128i k1, k2, k3, k4;

	__m128i k;
	__m128i cmp_0, cmp_1, cmp_2, cmp_3;
	__m128i lvl_1_0, lvl_1_1, lvl_1_2, lvl_1_3;
	__m128i lvl_2_A_0, lvl_2_A_1, lvl_2_A_2, lvl_2_A_3;
	__m128i lvl_2_B_0, lvl_2_B_1, lvl_2_B_2, lvl_2_B_3;
	__m128i cmp_A_0, cmp_A_1, cmp_A_2, cmp_A_3;
	__m128i cmp_B_0, cmp_B_1, cmp_B_2, cmp_B_3;
	__m128 cmp0, cmp1, cmp2, cmp3;
	// int res1 = 0, res2, res3;
	//level x keys result array
	int i;

	for (i = 0 ; i < num_probes ; i+=4) {
	//Loads 4 probes
	k = _mm_load_si128((__m128i*) &probes[i]);
	k1 = _mm_shuffle_epi32(k, _MM_SHUFFLE(0,0,0,0));
	k2 = _mm_shuffle_epi32(k, _MM_SHUFFLE(1,1,1,1));
	k3 = _mm_shuffle_epi32(k, _MM_SHUFFLE(2,2,2,2));
	k4 = _mm_shuffle_epi32(k, _MM_SHUFFLE(3,3,3,3));
	int res[3][4] = {{0}};

	//9 fanout root
	cmp_A_0 = _mm_cmpgt_epi32(k1, root1);
	cmp_B_0 = _mm_cmpgt_epi32(k1, root2);

	cmp_A_1 = _mm_cmpgt_epi32(k2, root1);
	cmp_B_1 = _mm_cmpgt_epi32(k2, root2);

	cmp_A_2 = _mm_cmpgt_epi32(k3, root1);
	cmp_B_2 = _mm_cmpgt_epi32(k3, root2);

	cmp_A_3 = _mm_cmpgt_epi32(k4, root1);
	cmp_B_3 = _mm_cmpgt_epi32(k4, root2);


	cmp_0 = _mm_packs_epi32(cmp_A_0, cmp_B_0);
	cmp_1 = _mm_packs_epi32(cmp_A_1, cmp_B_1);
	cmp_2 = _mm_packs_epi32(cmp_A_2, cmp_B_2);
	cmp_3 = _mm_packs_epi32(cmp_A_3, cmp_B_3);

	cmp_0 = _mm_packs_epi16(cmp_0, _mm_setzero_si128());
	cmp_1 = _mm_packs_epi16(cmp_1, _mm_setzero_si128());
	cmp_2 = _mm_packs_epi16(cmp_2, _mm_setzero_si128());
	cmp_3 = _mm_packs_epi16(cmp_3, _mm_setzero_si128());


	res[0][0] = _mm_movemask_epi8(cmp_0);
	res[0][1] = _mm_movemask_epi8(cmp_1);
	res[0][2] = _mm_movemask_epi8(cmp_2);
	res[0][3] = _mm_movemask_epi8(cmp_3);

	res[0][0] = _bit_scan_forward(res[0][0] ^ 0x1FFFF);
	res[0][1] = _bit_scan_forward(res[0][1] ^ 0x1FFFF);
	res[0][2] = _bit_scan_forward(res[0][2] ^ 0x1FFFF);
	res[0][3] = _bit_scan_forward(res[0][3] ^ 0x1FFFF);


	//5 fanout lvl 1
	lvl_1_0 = _mm_load_si128(( __m128i *) &index[1][res[0][0] << 2]);
	lvl_1_1 = _mm_load_si128(( __m128i *) &index[1][res[0][1] << 2]);
	lvl_1_2 = _mm_load_si128(( __m128i *) &index[1][res[0][2] << 2]);
	lvl_1_3 = _mm_load_si128(( __m128i *) &index[1][res[0][3] << 2]);

	cmp0 = _mm_castsi128_ps(_mm_cmpgt_epi32(k1, lvl_1_0));
	cmp1 = _mm_castsi128_ps(_mm_cmpgt_epi32(k2, lvl_1_1));
	cmp2 = _mm_castsi128_ps(_mm_cmpgt_epi32(k3, lvl_1_2));
	cmp3 = _mm_castsi128_ps(_mm_cmpgt_epi32(k4, lvl_1_3));

	res[1][0] = _mm_movemask_ps(cmp0);   // ps: epi32
	res[1][1] = _mm_movemask_ps(cmp1);
	res[1][2] = _mm_movemask_ps(cmp2);
	res[1][3] = _mm_movemask_ps(cmp3);


	res[1][0] = _bit_scan_forward(res[1][0] ^ 0x1FF);
	res[1][1] = _bit_scan_forward(res[1][1] ^ 0x1FF);
	res[1][2] = _bit_scan_forward(res[1][2] ^ 0x1FF);
	res[1][3] = _bit_scan_forward(res[1][3] ^ 0x1FF);


	res[1][0] += (res[0][0] << 2) + res[0][0];
	res[1][1] += (res[0][1] << 2) + res[0][1];
	res[1][2] += (res[0][2] << 2) + res[0][2];
	res[1][3] += (res[0][3] << 2) + res[0][3];


	//9 fanout lvl 2
	__m128i lvl_2_A_0 = _mm_load_si128(( __m128i *) &index[2][res[1][0] << 3]);
	__m128i lvl_2_B_0 = _mm_load_si128(( __m128i *) &index[2][(res[1][0] << 3) + 4]);

	__m128i lvl_2_A_1 = _mm_load_si128(( __m128i *) &index[2][res[1][1] << 3]);
	__m128i lvl_2_B_1 = _mm_load_si128(( __m128i *) &index[2][(res[1][1] << 3) + 4]);

	__m128i lvl_2_A_2 = _mm_load_si128(( __m128i *) &index[2][res[1][2] << 3]);
	__m128i lvl_2_B_2 = _mm_load_si128(( __m128i *) &index[2][(res[1][2] << 3) + 4]);

	__m128i lvl_2_A_3 = _mm_load_si128(( __m128i *) &index[2][res[1][3] << 3]);
	__m128i lvl_2_B_3 = _mm_load_si128(( __m128i *) &index[2][(res[1][3] << 3) + 4]);


	cmp_A_0 = _mm_cmpgt_epi32(k1, lvl_2_A_0);
	cmp_B_0 = _mm_cmpgt_epi32(k1, lvl_2_B_0);

	cmp_A_1 = _mm_cmpgt_epi32(k2, lvl_2_A_1);
	cmp_B_1 = _mm_cmpgt_epi32(k2, lvl_2_B_1);

	cmp_A_2 = _mm_cmpgt_epi32(k3, lvl_2_A_2);
	cmp_B_2 = _mm_cmpgt_epi32(k3, lvl_2_B_2);

	cmp_A_3 = _mm_cmpgt_epi32(k4, lvl_2_A_3);
	cmp_B_3 = _mm_cmpgt_epi32(k4, lvl_2_B_3);


	cmp_0 = _mm_packs_epi32(cmp_A_0, cmp_B_0);
	cmp_1 = _mm_packs_epi32(cmp_A_1, cmp_B_1);
	cmp_2 = _mm_packs_epi32(cmp_A_2, cmp_B_2);
	cmp_3 = _mm_packs_epi32(cmp_A_3, cmp_B_3);

	cmp_0 = _mm_packs_epi16(cmp_0, _mm_setzero_si128());
	cmp_1 = _mm_packs_epi16(cmp_1, _mm_setzero_si128());
	cmp_2 = _mm_packs_epi16(cmp_2, _mm_setzero_si128());
	cmp_3 = _mm_packs_epi16(cmp_3, _mm_setzero_si128());

	res[2][0] = _mm_movemask_epi8(cmp_0);
	res[2][1] = _mm_movemask_epi8(cmp_1);
	res[2][2] = _mm_movemask_epi8(cmp_2);
	res[2][3] = _mm_movemask_epi8(cmp_3);

	res[2][0] = _bit_scan_forward(res[2][0] ^ 0x1FFFF);
	res[2][1] = _bit_scan_forward(res[2][1] ^ 0x1FFFF);
	res[2][2] = _bit_scan_forward(res[2][2] ^ 0x1FFFF);
	res[2][3] = _bit_scan_forward(res[2][3] ^ 0x1FFFF);

	res[2][0] += (res[1][0] << 3) + res[1][0];
	res[2][1] += (res[1][1] << 3) + res[1][1];
	res[2][2] += (res[1][2] << 3) + res[1][2];
	res[2][3] += (res[1][3] << 3) + res[1][3];

	outputs[i] = res[2][0];
	outputs[i+1] = res[2][1];
	outputs[i+2] = res[2][2];
	outputs[i+3] = res[2][3];
	}

}

int main(int argc, const char * argv[]) {
    // insert code here...
    int i=0, j=0, total_keys=0;
    int debug = 1;
    int num_keys, num_probes, num_levels;
    //int maximum = INT_MAX;

    if(argc > 3) {
        num_keys = atoi(argv[1]);
        num_probes = atoi(argv[2]);
    } else {
        printf("Error!!\n");
        printf("Code should be invoked as build K P <levels>\n");
        printf("Where K is the number of keys used to build the tree, and P is the number of probes to perform\n");
        exit(0);
    }

    num_levels = argc - 3;
    uint32_t *branches = calloc(num_levels, sizeof(uint32_t)); // Fanout per level
    uint32_t *max_nodes = calloc(num_levels, sizeof(uint32_t)); //Max_nodes per level

    max_nodes[0] = 1;
    for (i = 0 ; i < num_levels ; i++) {
        branches[i] = atoi(argv[i+3]);
    }
    for (i = 1 ; i < num_levels ; i++) {
        max_nodes[i] = max_nodes[i-1] * branches[i-1]; 
    }
    for (i = 0 ; i < num_levels ; i++) {
        total_keys+=max_nodes[i]*(branches[i]-1); //Max total keys that can be stored in the index
    }
    if(num_keys > total_keys) {
        printf("Error!!\n");
        printf("Maximum number of keys that can be stored in such a tree = %d\n", total_keys);
        exit(0);
    }
    if(num_keys <= total_keys/branches[0]) {
        printf("Error!!\n");
        printf("Minimum number of keys that can be stored in such a tree = %d\n", (total_keys/branches[0]+1));
        exit(0);
    }

    //Allocate space for the index
    int32_t** index = (int32_t**)malloc(num_levels * sizeof(int32_t *));
    void *v_ptr;
    for (i = 0 ; i < num_levels ; i++) {
        if(posix_memalign(&v_ptr, 16, sizeof(int32_t) * max_nodes[i] * (branches[i] - 1)) != 0){
            printf("Error!!\n");
            printf("Unable to allocate memory for the index at level %d\n", i);
            exit(0);
        }
        index[i] = v_ptr;
    }

    //Keys per node at a level = branches[level] - 1
    //Generate the keys
    rand32_t *gen1 = rand32_init((uint32_t)time(NULL));
    int32_t *keys = generate_sorted_unique(num_keys, gen1);
    // free(gen1);
    // int keys[] = {11,22,33,44,55,66,77,88,99,110,121,132};
    // int keys[] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90};
    // int keys[] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64};
    // printf("The keys generated = \n");
    // for (i = 0 ; i < num_keys ; ++i) {
    //     printf("%d, ",keys[i]);
    // }
    // printf("\n\n");

    int32_t *filled = calloc(num_levels, sizeof(int32_t));
    int32_t *gone_up = calloc(num_levels, sizeof(int32_t));

    //Creating the index
    for(i = 0; i < num_keys; i++) {
        // printf("Creating index\n");
        fill_up(num_levels-1, branches, keys[i], filled, index, gone_up);
    }

    free(gone_up);

    // Fill empty space in tree with INT_MAX
    for (i = 0 ; i < num_levels ; i++) {
        for(j = filled[i]; j < max_nodes[i] * (branches[i] - 1); j++) {
            index[i][j] = INT_MAX;
        }
    }
    //Print the tree
    // print_tree(index, max_nodes, branches, num_levels);


    //Generate the probes
    // rand32_t *gen2 = rand32_init((uint32_t)time(NULL));
    int32_t *probes = generate(num_probes, gen1);
    free(gen1);
    // int probes[] = {2147483647, 64, 66, 11, -98, 2};
    // printf("\nThe probes generated = \n");
    // for (i = 0 ; i < num_probes ; ++i) {
    //     printf("%d, ",probes[i]);
    // }
    // printf("\n\n");


    //Peform the search
    int *outputs_Bin = calloc(num_probes, sizeof(int));
    int *outputs_SSE = calloc(num_probes, sizeof(int));

    clock_t begin, end;
	// double time_spent;

	begin = clock();
    for (i = 0 ; i < num_probes ; i++) {
        //Search this probe
        int match_prev = 0;
        for (j = 0 ; j < num_levels ; j++) {
            //Next jump or offset = match_prev * (number of keys in a node)
            //branches[j]-1 = (number of keys in a node)
            int low = match_prev * (branches[j]-1);
            match_prev = bin_Search(index[j], low, low + branches[j] - 2, probes[i]); //-2!!
            // printf("Bin Match - %d\n", match_prev);
            //low/(branches[j]-1) = Number of nodes lesser than the node containing the match in the parent level
            match_prev += low/(branches[j]-1);
        }
        outputs_Bin[i] = match_prev;
    }
    end = clock();
    double time_spent_Bin = (double)(end - begin) / CLOCKS_PER_SEC;
    // printf("The probe - range_id array from Binary Search is:-\n");
    // for (i = 0 ; i < num_probes ; i++) {
    //     printf("%d %d\n", probes[i], outputs_Bin[i]);
    // }
    // printf("\n");

    begin = clock();
    for (i = 0 ; i < num_probes ; i++) {
        //Search this probe
        int match_prev = 0;
        for (j = 0 ; j < num_levels ; j++) {
            match_prev = SSE_Search(index[j], match_prev, branches[j], probes[i]);
        }
        outputs_SSE[i] = match_prev;
    }
    end = clock();

    double time_spent_SSE = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("The probe - range_id array from SSE Search is:-\n");
    for (i = 0 ; i < num_probes ; i++) {
        printf("%d %d\n", probes[i], outputs_SSE[i]);
    }
    printf("\n");

    printf("Time taken by Binary Search method = %f\n", time_spent_Bin);
    printf("Time taken by SIMD method = %f\n", time_spent_SSE);

    if (num_levels == 3 && branches[0] == 9 && branches[1] == 5 && branches[2] == 9) {
    	int *outputs_HC = calloc(num_probes, sizeof(int));
    	begin = clock();
    	Search_9_5_9(index, probes, num_probes, outputs_HC);
    	end = clock();
    	double time_spent_HC = (double)(end - begin) / CLOCKS_PER_SEC;
    	printf("Time taken by Hard-Coded SIMD method = %f\n", time_spent_HC);
    	// printf("The probe - range_id array from Hard Coded Search is:-\n");
    	// for (i = 0 ; i < num_probes ; i++) {
     //    	printf("%d %d\n", probes[i], outputs_HC[i]);
    	// }
    	// printf("\n");
    	free(outputs_HC);
    }

    free(branches);
    free(max_nodes);
    //TODO!!
    free(probes);
    free(outputs_Bin);
    free(outputs_SSE);
    free(keys);
    for (j = 0 ; j < num_levels ; j++) {
        free(index[j]);
    }
    free(index);
    free(filled);
    return 0;
}
