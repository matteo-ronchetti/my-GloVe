#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>

using namespace std;

struct MatElement{
   unsigned i;
   unsigned j;
   real v;
};

class SparseMat{
public:
   std::unordered_map<unsigned long,real> mymap;

   real& operator() (unsigned i, unsigned j){
      return this->at(i,j);
   }

   real& at(unsigned i, unsigned j){
      return mymap[((unsigned long)(i) << 32) + (unsigned long)(j)];
   }

   vector<MatElement> get_element_list(){
      vector<MatElement> res;
      for ( auto it = mymap.cbegin(); it != mymap.cend(); ++it ){
         MatElement el;
         el.i = it->first >> 32;
         el.j = it->first;
         el.v = it->second;
         if(el.i && el.j)res.push_back(el);
      }
      return res;
   }

   vector<MatElement> get_element_list_with_map(const vector<unsigned>& map) const{
      vector<MatElement> res;
      for ( auto it = mymap.cbegin(); it != mymap.cend(); ++it ){
         MatElement el;
         el.i = map[unsigned(it->first >> 32)];
         el.j = map[unsigned(it->first)];
         el.v = it->second;
         if(el.i != numeric_limits<unsigned>::max() && el.j != numeric_limits<unsigned>::max()){
            res.push_back(el);
         }

      }
      return res;
   }

   void save(string path){
      auto list = this->get_element_list();

      save_vector(path, list);
   }

   void save_with_map(string path, const vector<unsigned>& map){
      auto list = this->get_element_list_with_map(map);

      ofstream fout(path, ios::out | ios::binary);
      fout.write((char*)&list[0], list.size() * sizeof(MatElement));
      fout.close();
   }

   SparseMat& operator += (SparseMat& o){
      for ( auto it = o.mymap.cbegin(); it != o.mymap.cend(); ++it ){
         mymap[it->first] += it->second;
      }

      return *this;
   }

   unsigned count(){
      return this->mymap.size();
   }

   real sum(){
      real res = 0;
      for ( auto it = mymap.cbegin(); it != mymap.cend(); ++it ){
          res += it->second;
      }
      return res;
   }

   void simmetrify(){
      for ( auto it = mymap.cbegin(); it != mymap.cend(); ++it ){
         unsigned i = it->first >> 32;
         unsigned j = it->first;
         real v = it->second;
         this->at(j,i) += v;
      }
   }
};
