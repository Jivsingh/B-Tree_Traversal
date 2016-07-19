#ifndef rand_func_h
#define rand_func_h


#endif /* rand_func_h */

typedef struct {
    size_t index;
    uint32_t num[625];
} rand32_t;

rand32_t *rand32_init(uint32_t x);
uint32_t rand32_next(rand32_t *s);
int int32_cmp(const void *x, const void *y);
int32_t *generate(size_t n, rand32_t *gen);
int32_t *generate_sorted_unique(size_t n, rand32_t *gen);
void ratio_per_bit(const int32_t *a, size_t n);




