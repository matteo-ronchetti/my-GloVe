#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>

using namespace std;

class Dictionary{

public:
   unordered_map<string, pair<unsigned,unsigned> > dict;
   unsigned count = 1;

   Dictionary(){}

   Dictionary(unordered_map<string, pair<unsigned,unsigned> >& _dict):dict(_dict){
      count = dict.size();
   }

   Dictionary(string path){
      ifstream file(path);

      string word;
      unsigned index;
      unsigned frequency;

      while (file >> word >> index >> frequency){
          dict[word] = make_pair(index, frequency);
          this->count++;
       }
   }

   int add_file(string path){
      ifstream file;
      file.open (path);
      if (!file.is_open()) return 0;

      string word;
      while (file >> word){
           if(dict.count(word) > 0){
             dict[word].second++;
          }else{
             dict[word] = make_pair(this->count, 1);
             this->count++;
          }
      }

      return this->count-1;
   }

   int filter_by_frequency(unsigned min_freq){
      count = 1;
      for(auto it = dict.begin(); it != dict.end(); ++it ){
         if(it->second.second >= min_freq){
            it->second.first = count; //set index to next filtered index
            count++;
         }else{
            it->second.first = 0; //set index to null index
         }
      }
      return count - 1;
   }

   int compute_document_vector(vector<unsigned>& document, const string& path, int window_size){
      vector<char> buff;
      buff.reserve(128);

      ifstream fin(path);
      if (!fin.is_open()) return 0;

      char tmp;
      while (fin >> noskipws >> tmp) {
          if(tmp == ' ' || tmp == '\n'){
             document.push_back(dict[string(&buff[0])].first);
             buff.clear();
             if(tmp == '\n'){
                document.resize(document.size()+window_size, 0);
             }
          }else{
             buff.push_back(tmp);
          }
      }

      return document.size();
   }

   void save(const string& path){
      ofstream file;
      file.open(path);

      for(auto it = dict.cbegin(); it != dict.cend(); ++it ){
         if(it->second.first){
            file << it->first << " " << it->second.first << " " << it->second.second << endl;
         }
      }

      file.close();
   }

   // void save_with_map(string path, const vector<unsigned>& map){
   //    ofstream file;
   //    file.open(path);
   //
   //    for(auto it = dict.cbegin(); it != dict.cend(); ++it ){
   //       unsigned index = map[it->second.first];
   //       if(index != numeric_limits<unsigned>::max()){
   //          file << it->first << " " << index << " " << it->second.second << endl;
   //       }
   //    }
   //
   //    file.close();
   // }

   // vector<unsigned> map_file(string path){
   //    vector<unsigned>res;
   //
   //    ifstream file;
   //    file.open (path);
   //    if (!file.is_open()) return res;
   //
   //    string word;
   //    while (file >> word){
   //       res.push_back(dict[word].first);
   //    }
   //
   //    return res;
   // }

   // Dictionary filter_by_frequency(unsigned min_freq){
   //    Dictionary res;
   //    for(auto it = dict.cbegin(); it != dict.cend(); ++it )if(it->second.second >= min_freq){
   //       res.dict[it->first] = it->second;
   //    }
   //    return res;
   // }
   //
   // std::vector<unsigned> get_filtered_shuffled_map(unsigned min_freq) const{
   //    unsigned count = 0;
   //    for(auto it = dict.cbegin(); it != dict.cend(); ++it )if(it->second.second >= min_freq){
   //       count++;
   //    }
   //
   //    std::vector<unsigned> mini_index_map;
   //    for(unsigned i = 0; i < count; i++)mini_index_map.push_back(i);
   //    random_shuffle(mini_index_map.begin(), mini_index_map.end());
   //
   //    std::vector<unsigned> index_map;
   //    index_map.resize(dict.size());
   //
   //    count = 0;
   //    for(auto it = dict.cbegin(); it != dict.cend(); ++it ){
   //       if(it->second.second >= min_freq){
   //          index_map[it->second.first] = mini_index_map[count];
   //          count++;
   //       }else{
   //          index_map[it->second.first] = numeric_limits<unsigned>::max();
   //       }
   //    }
   //
   //    return index_map;
   // }
};

// const std::string ws = " \t\r\n";
// std::size_t pos = 0;
// while (pos != s.size()) {
//     std::size_t from = s.find_first_not_of(ws, pos);
//     if (from == std::string::npos) {
//         break;
//     }
//     std::size_t to = s.find_first_of(ws, from+1);
//     if (to == std::string::npos) {
//         to = s.size();
//     }
//     // If you want an individual word, copy it with substr.
//     // The code below simply prints it character-by-character:
//     std::cout << "'";
//     for (std::size_t i = from ; i != to ; i++) {
//         std::cout << s[i];
//     }
//     std::cout << "'" << std::endl;
//     pos = to;
// }

// int main(){
//    Dictionary dict;
//    dict.add_file("text");
//    dict.save("dict");
// }
