#include <immintrin.h>


constexpr bool USE_AVX = true;

void random_normal(real* X, int count, real stdev = 1){
   std::random_device rd;
   std::mt19937 gen(rd());

   normal_distribution<real> normal(0, stdev);

   for(int i = 0;i < count; i++){
      X[i] = normal(gen);
   }
}

template<class T> void zeros(T* X, int count){
   for(int i = 0;i < count; i++){
      X[i] = 0;
   }
}

inline real check_nan(real update) {
    if (isnan(update) || isinf(update)) {
        fprintf(stderr,"\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

inline void* aligned_malloc(size_t align, size_t size) {
    void *result;
    #ifdef _MSC_VER
    result = _aligned_malloc(size, align);
    #else
     if(posix_memalign(&result, align, size)) result = 0;
    #endif
    return result;
}

static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}


void adam_thread(unsigned tid, unsigned num_threads, unsigned embedding_size, const std::vector<MatElement>& cooccurrences, const real eta, const real x_max, const real learn_rate_decay, const real mu, const real nu, real *words, real *first_moment, real *second_moment, real *biases, real *bias_first_moment, real *bias_second_moment, real *cost, unsigned *update_count){
   //real* updates_1 = (real*)aligned_malloc(32, embedding_size*sizeof(real));//new real[embedding_size];
   //real* updates_2 = (real*)aligned_malloc(32, embedding_size*sizeof(real));//new real[embedding_size];

   unsigned end = cooccurrences.size();
   cost[tid] = 0;

   __m256 reg1,reg2,reg3,reg4,reg5,reg6,reg7;

   __m256 reg_mu = _mm256_set1_ps(mu);
   __m256 reg_nu = _mm256_set1_ps(nu);
   __m256 reg_nmu = _mm256_set1_ps(1 - mu);
   __m256 reg_nnu = _mm256_set1_ps(1 - nu);


   for(unsigned i = tid; i < end; i += num_threads){
      const unsigned w1 = cooccurrences[i].i;
      const unsigned w2 = cooccurrences[i].j;// + dict_size;
      const real x = cooccurrences[i].v;

      // if(w1 > 71290 || w2 > 2*71290){
      //    cout << "Out of range" << endl;
      //    return;
      // }

      const unsigned l1 = w1*embedding_size;
      const unsigned l2 = w2*embedding_size;


      // if( x <= 0 ){
      //    cerr << x << endl;
      //    continue;
      // }

      real E = biases[w1] + biases[w2] - log(x);

      if(USE_AVX){
         reg1 = _mm256_setzero_ps();
         for(int b = 0; b < embedding_size; b += 8){
            reg2 = _mm256_load_ps(&words[l1+b]);
            reg3 = _mm256_load_ps(&words[l2+b]);
            reg1 = _mm256_fmadd_ps(reg2, reg3, reg1);
         }
         E += _mm256_reduce_add_ps(reg1);
      }else{
         for(int b = 0; b < embedding_size; b++)E += words[l1 + b] * words[l2 + b]; // takes 440ms
      }

      real weighted_error = (x > x_max) ? E : pow(x / x_max, alpha)*E;

      cost[tid] += 0.5 * weighted_error * E;

      update_count[w1]++;
      update_count[w2]++;

      real learn_rate_1 = eta/(1.0f + learn_rate_decay*update_count[w1]) * sqrt(1 - pow(nu,update_count[w1]))/(1 - pow(mu, update_count[w1]));
      real learn_rate_2 = eta/(1.0f + learn_rate_decay*update_count[w2]) * sqrt(1 - pow(nu,update_count[w2]))/(1 - pow(mu, update_count[w2]));

      // if(isnan(learn_rate_1) || isinf(learn_rate_1)){
      //    cerr << "Learning rate\n";
      //    return;
      // }

      if(USE_AVX){
         reg1 = _mm256_set1_ps(weighted_error);
         __m256 reg_lr1 = _mm256_set1_ps(learn_rate_1);
         __m256 reg_lr2 = _mm256_set1_ps(learn_rate_2);

         for(unsigned b = 0; b < embedding_size; b += 8) {
            reg5 = _mm256_load_ps(&words[l1+b]);
            reg6 = _mm256_load_ps(&words[l2+b]);
            reg7 = _mm256_load_ps(&words[l1+b]);


            //temp1
            reg2 = _mm256_mul_ps(reg1, reg6);
            //first_moment 1
            reg3 = _mm256_load_ps(&first_moment[l1+b]);
            reg3 = _mm256_fmadd_ps(reg_mu, reg3, _mm256_mul_ps(reg_nmu, reg2));
            //second_moment 1
            reg4 = _mm256_load_ps(&second_moment[l1+b]);
            reg4 = _mm256_fmadd_ps(reg_nu, reg4, _mm256_mul_ps(reg_nnu, _mm256_mul_ps(reg2,reg2)));
            //update word
            reg7 = _mm256_sub_ps(reg5, _mm256_mul_ps(reg_lr1,_mm256_mul_ps(reg3, _mm256_rsqrt_ps(reg4))));

            _mm256_stream_ps(&first_moment[l1+b], reg3);
            _mm256_stream_ps(&second_moment[l1+b], reg4);


            //temp2
            reg2 = _mm256_mul_ps(reg1, reg5);
            //first_moment 2
            reg3 = _mm256_load_ps(&first_moment[l2+b]);
            reg3 = _mm256_fmadd_ps(reg_mu, reg3, _mm256_mul_ps(reg_nmu, reg2));
            //second_moment 2
            reg4 = _mm256_load_ps(&second_moment[l2+b]);
            reg4 = _mm256_fmadd_ps(reg_nu, reg4, _mm256_mul_ps(reg_nnu, _mm256_mul_ps(reg2,reg2)));
            //update word
            reg6 = _mm256_sub_ps(reg6, _mm256_mul_ps(reg_lr2,_mm256_mul_ps(reg3, _mm256_rsqrt_ps(reg4))));

            _mm256_stream_ps(&first_moment[l2+b], reg3);
            _mm256_stream_ps(&second_moment[l2+b], reg4);

            _mm256_stream_ps(&words[l1+b], reg7);
            _mm256_stream_ps(&words[l2+b], reg6);

         }
      }else{
         for (unsigned b = 0; b < embedding_size; b++) {
             const real temp1 = weighted_error * words[l2 + b];
             const real temp2 = weighted_error * words[l1 + b];

             // compute moments ( takes 750ms )
             first_moment[b + l1] = mu*first_moment[b + l1] + (1-mu)*temp1;
             first_moment[b + l2] = mu*first_moment[b + l2] + (1-mu)*temp2;

             second_moment[b + l1] = nu*second_moment[b + l1] + (1-nu)*temp1 * temp1;
             second_moment[b + l2] = nu*second_moment[b + l2] + (1-nu)*temp2 * temp2;

             words[b + l1] -= learn_rate_1 * first_moment[b + l1] /(sqrt(second_moment[b + l1]) + eps);
             words[b + l2] -= learn_rate_2 * first_moment[b + l2] /(sqrt(second_moment[b + l2]) + eps);
         }
      }


      //
      bias_first_moment[w1] = mu*bias_first_moment[w1] + (1-mu)*weighted_error;
      bias_first_moment[w2] = mu*bias_first_moment[w2] + (1-mu)*weighted_error;

      bias_second_moment[w1] = nu*bias_second_moment[w1] + (1-nu)*weighted_error*weighted_error;
      bias_second_moment[w2] = nu*bias_second_moment[w2] + (1-nu)*weighted_error*weighted_error;

      // updates for bias terms
      biases[w1] -= learn_rate_1 * bias_first_moment[w1] /(sqrt(bias_second_moment[w1]) + eps);
      biases[w2] -= learn_rate_2 * bias_first_moment[w2] /(sqrt(bias_second_moment[w2]) + eps);

      if(isnan(biases[w1]) || isinf(biases[w1])){
         cerr << "got NaN\n";
         return;
      }
   }

   //free(updates_1);
   //free(updates_2);
}


class Embeddings{
   real *words, *first_moment, *second_moment;
   real *biases, *bias_first_moment, *bias_second_moment;
   real *cost;

   unsigned *update_count;

   unsigned embedding_size;

   unsigned num_threads;
   std::thread* threads;

public:
   bool allocated = false;

   Embeddings(unsigned _embedding_size, int _num_threads):embedding_size(_embedding_size),num_threads(_num_threads){
      threads = new std::thread[num_threads];
   }

   void allocate(int dict_size){
      words = (real*) aligned_malloc(32, dict_size*embedding_size*sizeof(real)); //new real[dict_size*embedding_size];
      first_moment = (real*) aligned_malloc(32, dict_size*embedding_size*sizeof(real));
      second_moment = (real*) aligned_malloc(32, dict_size*embedding_size*sizeof(real));

      biases = new real[dict_size];
      bias_first_moment = new real[dict_size];
      bias_second_moment = new real[dict_size];

      cost = new real[num_threads];

      update_count = new unsigned[dict_size];

      random_normal(words, dict_size*embedding_size, 0.5/embedding_size);
      zeros(first_moment, dict_size*embedding_size);
      zeros(second_moment, dict_size*embedding_size);

      zeros(biases, dict_size);
      zeros(bias_first_moment, dict_size);
      zeros(bias_second_moment, dict_size);

      zeros(update_count, dict_size);

      allocated = true;
   }

   real train_iteration(const std::vector<MatElement>& cooccurrences, const real& eta, const real& x_max, const real& learn_rate_decay, const real& mu, const real& nu){
      unsigned end = cooccurrences.size();
      for(unsigned j = 0; j < num_threads; ++j){
         //threads[j] = std::thread(adam_thread, j, num_threads, embedding_size, ref(cooccurrences), eta, x_max, decay, mu, nu, words, first_moment, second_moment, biases, bias_first_moment, bias_second_moment, cost);
         threads[j] = std::thread([&](unsigned tid){
            cost[tid] = 0;

            __m256 reg1,reg2,reg3,reg4,reg5,reg6,reg7;

            __m256 reg_mu = _mm256_set1_ps(mu);
            __m256 reg_nu = _mm256_set1_ps(nu);
            __m256 reg_nmu = _mm256_set1_ps(1 - mu);
            __m256 reg_nnu = _mm256_set1_ps(1 - nu);

            for(unsigned i = tid; i < end; i += num_threads){
               const unsigned w1 = cooccurrences[i].i;
               const unsigned w2 = cooccurrences[i].j;// + dict_size;
               const real x = cooccurrences[i].v;

               const unsigned l1 = w1*embedding_size;
               const unsigned l2 = w2*embedding_size;

               real E = biases[w1] + biases[w2] - log(x);

               if(USE_AVX){
                  reg1 = _mm256_setzero_ps();
                  for(int b = 0; b < embedding_size; b += 8){
                     reg2 = _mm256_load_ps(&words[l1+b]);
                     reg3 = _mm256_load_ps(&words[l2+b]);
                     reg1 = _mm256_fmadd_ps(reg2, reg3, reg1);
                  }
                  E += _mm256_reduce_add_ps(reg1);
               }else{
                  for(int b = 0; b < embedding_size; b++)E += words[l1 + b] * words[l2 + b]; // takes 440ms
               }

               real weighted_error = (x > x_max) ? E : pow(x / x_max, alpha)*E;

               cost[tid] += 0.5 * weighted_error * E;

               update_count[w1]++;
               update_count[w2]++;

               real learn_rate_1 = eta/(1.0f + learn_rate_decay*update_count[w1]) * sqrt(1 - pow(nu,update_count[w1]))/(1 - pow(mu, update_count[w1]));
               real learn_rate_2 = eta/(1.0f + learn_rate_decay*update_count[w2]) * sqrt(1 - pow(nu,update_count[w2]))/(1 - pow(mu, update_count[w2]));

               if(USE_AVX){
                  reg1 = _mm256_set1_ps(weighted_error);
                  __m256 reg_lr1 = _mm256_set1_ps(learn_rate_1);
                  __m256 reg_lr2 = _mm256_set1_ps(learn_rate_2);

                  for(unsigned b = 0; b < embedding_size; b += 8) {
                     reg5 = _mm256_load_ps(&words[l1+b]);
                     reg6 = _mm256_load_ps(&words[l2+b]);
                     reg7 = _mm256_load_ps(&words[l1+b]);

                     //temp1
                     reg2 = _mm256_mul_ps(reg1, reg6);
                     //first_moment 1
                     reg3 = _mm256_load_ps(&first_moment[l1+b]);
                     reg3 = _mm256_fmadd_ps(reg_mu, reg3, _mm256_mul_ps(reg_nmu, reg2));
                     //second_moment 1
                     reg4 = _mm256_load_ps(&second_moment[l1+b]);
                     reg4 = _mm256_fmadd_ps(reg_nu, reg4, _mm256_mul_ps(reg_nnu, _mm256_mul_ps(reg2,reg2)));
                     //update word
                     reg7 = _mm256_sub_ps(reg5, _mm256_mul_ps(reg_lr1,_mm256_mul_ps(reg3, _mm256_rsqrt_ps(reg4))));

                     _mm256_stream_ps(&first_moment[l1+b], reg3);
                     _mm256_stream_ps(&second_moment[l1+b], reg4);


                     //temp2
                     reg2 = _mm256_mul_ps(reg1, reg5);
                     //first_moment 2
                     reg3 = _mm256_load_ps(&first_moment[l2+b]);
                     reg3 = _mm256_fmadd_ps(reg_mu, reg3, _mm256_mul_ps(reg_nmu, reg2));
                     //second_moment 2
                     reg4 = _mm256_load_ps(&second_moment[l2+b]);
                     reg4 = _mm256_fmadd_ps(reg_nu, reg4, _mm256_mul_ps(reg_nnu, _mm256_mul_ps(reg2,reg2)));
                     //update word
                     reg6 = _mm256_sub_ps(reg6, _mm256_mul_ps(reg_lr2,_mm256_mul_ps(reg3, _mm256_rsqrt_ps(reg4))));

                     _mm256_stream_ps(&first_moment[l2+b], reg3);
                     _mm256_stream_ps(&second_moment[l2+b], reg4);

                     _mm256_stream_ps(&words[l1+b], reg7);
                     _mm256_stream_ps(&words[l2+b], reg6);

                  }
               }else{
                  for (unsigned b = 0; b < embedding_size; b++) {
                      const real temp1 = weighted_error * words[l2 + b];
                      const real temp2 = weighted_error * words[l1 + b];

                      // compute moments ( takes 750ms )
                      first_moment[b + l1] = mu*first_moment[b + l1] + (1-mu)*temp1;
                      first_moment[b + l2] = mu*first_moment[b + l2] + (1-mu)*temp2;

                      second_moment[b + l1] = nu*second_moment[b + l1] + (1-nu)*temp1 * temp1;
                      second_moment[b + l2] = nu*second_moment[b + l2] + (1-nu)*temp2 * temp2;

                      words[b + l1] -= learn_rate_1 * first_moment[b + l1] /(sqrt(second_moment[b + l1]) + eps);
                      words[b + l2] -= learn_rate_2 * first_moment[b + l2] /(sqrt(second_moment[b + l2]) + eps);
                  }
               }

               bias_first_moment[w1] = mu*bias_first_moment[w1] + (1-mu)*weighted_error;
               bias_first_moment[w2] = mu*bias_first_moment[w2] + (1-mu)*weighted_error;

               bias_second_moment[w1] = nu*bias_second_moment[w1] + (1-nu)*weighted_error*weighted_error;
               bias_second_moment[w2] = nu*bias_second_moment[w2] + (1-nu)*weighted_error*weighted_error;

               // updates for bias terms
               biases[w1] -= learn_rate_1 * bias_first_moment[w1] /(sqrt(bias_second_moment[w1]) + eps);
               biases[w2] -= learn_rate_2 * bias_first_moment[w2] /(sqrt(bias_second_moment[w2]) + eps);

               if(isnan(biases[w1]) || isinf(biases[w1])){
                  cerr << "got NaN\n";
                  return;
               }
            }
         }, j);
      }

      //join threads
      for(int j = 0; j < num_threads; ++j)threads[j].join();

      real total_cost = 0;
      for(int j = 0; j < num_threads; ++j)total_cost += cost[j];

      return total_cost/cooccurrences.size();
   }

   void save_embeddings(const Dictionary& dict, const string& path){
      ofstream file(path);
      for(auto it = dict.dict.cbegin(); it != dict.dict.cend(); ++it ){
         unsigned w = it->second.first;
         if(it->second.first){
             file << it->first << " ";

             for(int i = 0; i < embedding_size; i++){
                file << words[w*embedding_size+i];
                if(i == embedding_size-1){
                   file << endl;
                }else{
                   file << " ";
                }
             }
         }
         //file << biases[w] << endl;
      }
      file.close();
   }
};
