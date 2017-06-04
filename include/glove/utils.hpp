#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

int get_all_files(string dir, vector<string> &files){
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if (dirp->d_type == DT_REG) files.push_back(dir + "/" + string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

template<class T> void save_vector(string path, const vector<T>& vec){
   ofstream fout(path, ios::out | ios::binary);
   fout.write((char*)&vec[0], vec.size() * sizeof(T));
   fout.close();
}

template<class T> vector<T> load_vector(string path){
   ifstream file(path, ios::binary);

   auto fsize = file.tellg();
   file.seekg( 0, std::ios::end );
   fsize = file.tellg() - fsize;
   file.seekg(0, std::ios::beg);

   vector<T> res;
   res.resize(fsize/sizeof(T));

   file.read((char*)&res[0], res.size() * sizeof(T));
   file.close();

   return res;
}
