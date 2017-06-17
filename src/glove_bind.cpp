#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <thread>
#include <random>

using namespace std;

#include <glove/config.hpp>
#include <glove/utils.hpp>
#include <glove/dictionary.hpp>
#include <glove/sparse-mat.hpp>
#include <glove/embeddings.hpp>


void cooccurrence_count_thread(int j, const vector<unsigned>& document, SparseMat& cooccurrences, const real* win_weights){
   if(document.size() < j)return;
   for(int i = 0; i < document.size()-j; i++){
      unsigned a = document[i];
      unsigned b = document[i+j];
      if(a != b) cooccurrences(min(a,b), max(a,b)) += win_weights[j];
   }
}

class GloVe{
   Dictionary dict;
   vector<unsigned> document;
   std::vector<MatElement> cooccurrences;
    vector<real> window_weights;

   int num_threads;
   Embeddings embeddings;

public:
   GloVe(int _embedding_size, int _num_threads):num_threads(_num_threads),embeddings(_embedding_size, _num_threads){}

    void set_window_weights(vector<real> win_weights){
        this->window_weights = win_weights;
    }

   int add_file(const string& path){
      return dict.add_file(path);
   }

   int filter_dictionary(int min_word_frequency){
      return dict.filter_by_frequency(min_word_frequency);
   }

   void save_dictionary(const string& path){
      dict.save(path);
   }

   void load_dictionary(const string& path){
      dict = Dictionary(path);
   }

   void save_cooccurrences(const string& path){
      save_vector(path, cooccurrences);
   }

   void load_coocurrences(const string& path){
      cooccurrences = load_vector<MatElement>(path);
   }

   int compute_document_vector(const string& path, int window_size){
      return dict.compute_document_vector(document, path, window_size);
   }

   int compute_coocurrence(int window_size){
      std::thread* threads = new std::thread[num_threads];
      SparseMat* _cooccurrences = new SparseMat[num_threads];

      for(int j = 1; j <= window_size; j += num_threads){
         for(int i = 0; i < num_threads; i++)if(i+j <= window_size){
            threads[i] = std::thread(cooccurrence_count_thread, j+i, ref(document), ref(_cooccurrences[i]), &window_weights[0]);
         }
         for(int i = 0; i < num_threads; i++) if(threads[i].joinable())threads[i].join();
      }
      for(int i = 1; i < num_threads;i++){
         _cooccurrences[0] += _cooccurrences[i];
      }

      cooccurrences = _cooccurrences[0].get_element_list();
      document.clear();

      delete[] _cooccurrences;
      delete[] threads;


      return cooccurrences.size();
   }

   void shuffle_coocurrences(){
      random_shuffle(cooccurrences.begin(), cooccurrences.end());
   }

   real train_iteration(real eta, real x_max, real decay, real mu, real nu){
      if(! embeddings.allocated ){
         embeddings.allocate(dict.count);
      }
      return embeddings.train_iteration(cooccurrences, eta, x_max, decay, mu, nu);
   }

   void save_embeddings(const string& path){
      embeddings.save_embeddings(dict, path);
   }
};



PYBIND11_PLUGIN(_glove) {
    py::module m("_glove", "GloVe");

    py::class_<GloVe>(m, "_GloVe")
        .def(py::init<int,int>())
        .def("set_window_weights", &GloVe::set_window_weights)
        .def("add_file", &GloVe::add_file)
        .def("filter_dictionary", &GloVe::filter_dictionary)
        .def("save_dictionary", &GloVe::save_dictionary)
        .def("load_dictionary", &GloVe::load_dictionary)
        .def("save_cooccurrences", &GloVe::save_cooccurrences)
        .def("load_coocurrences", &GloVe::load_coocurrences)
        .def("compute_document_vector", &GloVe::compute_document_vector)
        .def("compute_coocurrence", &GloVe::compute_coocurrence)
        .def("shuffle_coocurrences", &GloVe::shuffle_coocurrences)
        .def("train_iteration", &GloVe::train_iteration)
        .def("save_embeddings", &GloVe::save_embeddings);

    return m.ptr();
}
