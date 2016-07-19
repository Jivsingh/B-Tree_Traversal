#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include "rand_func.h"

void fill_up(int level, uint32_t* branches, int32_t key, int32_t* filled, int32_t** index, int32_t* gone_up) {
    // if(level < 0) {
    //     printf("This probably shouldn't have happened. Check boundary!!\n");
    // }

    if((filled[level]% (branches[level]-1)==0) && !gone_up[level] && filled[level] != 0) {
        fill_up(level-1, branches, key, filled, index, gone_up);
        gone_up[level] = 1;
        // return 1;
    } else {
        // printf("Filled at level - %d\n", filled[level]);
        index[level][filled[level]] = key;
        // printf("Filled - %d\n", key);
        filled[level]++;
        gone_up[level] = 0;
        // return 0;
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
    while(low<=high) // = ??
    {
        int mid = (low+high)/2;
        if(arr[mid]==target && arr[mid] != INT_MAX)
            return mid;
        else if (arr[mid]>=target)
        {
            high = mid-1;
        } else
        {
            low = mid+1;
        }
    }

    return low;
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
    rand32_t *gen1 = rand32_init(time(NULL));
    int32_t *keys = generate_sorted_unique(num_keys, gen1);
    free(gen1);
    //TODO!!
    // int keys[] = {11,22,33,44,55,66,77,88,99,110,121,132};
    // int keys[] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90};
    // int keys[] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64};
    // int keys[] = {2,4,6,8,10,12,14,16,18};
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

    //TODO!!
    //Generate the probes
    rand32_t *gen2 = rand32_init(time(NULL));
    int32_t *probes = generate(num_probes, gen2);
    free(gen2);
    // int probes[] = {2147483647, 64, 66, 11, -98, 2};
    // printf("\nThe probes generated = \n");
    // for (i = 0 ; i < num_probes ; ++i) {
    //     printf("%d, ",probes[i]);
    // }
    // printf("\n\n");

    //Peform the search
    int *outputs = calloc(num_probes, sizeof(int));
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
        outputs[i] = match_prev;
    }

    printf("The probe - range_id array is:-\n");
    for (i = 0 ; i < num_probes ; i++) {
        printf("%d %d\n", probes[i], outputs[i]);
    }
    printf("\n\n");

    free(branches);
    free(max_nodes);
    //TODO!!
    free(probes);
    free(outputs);
    free(keys);
    for (j = 0 ; j < num_levels ; j++) {
        free(index[j]);
    }
    free(index);
    free(filled);
    return 0;
}
