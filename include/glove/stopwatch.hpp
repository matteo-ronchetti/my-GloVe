#ifndef STOPWATCH_HPP
#define STOPWATCH_HPP

#include <chrono>
#include <iostream>
#include <string>

using namespace std;

class Stopwatch{
	float time = 0; // total measured time
	int count = 0; // lap count
	bool started = false;
	string name;
	std::chrono::system_clock::time_point startTime;

    void _start(){
        startTime = std::chrono::system_clock::now();
        started = true;
    }

    void _stop(){
        count++;

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - startTime);
        //convert from nanoseconds to milliseconds
        time += elapsed.count()/1000000.0;
        started = false;
    }

public:
    friend std::ostream &operator<<(std::ostream &os, Stopwatch const &watch);


	Stopwatch(string n = ""):name(n){} //initialize with a name

	void tick(){
		if(started){
            _stop();
		}else{
            start();
		}
	}

    void start(){
        if(!started)_start();
    }

    void stop(){
        if(started)_stop();
    }

	Stopwatch operator + (const Stopwatch& o){
		Stopwatch res(name + " + " + o.name);
		res.time = time + o.time;
		res.count = count;
		return res;
	}

	float fps(){
		return (1000*count)/time;
	}
};

std::ostream &operator<<(std::ostream &os, Stopwatch const &watch) {
    return os << watch.name << ":\n     time: " << watch.time << "ms\n     count: " << watch.count << "\n     average time: " << watch.time/watch.count << "ms" << endl;
}

#endif
