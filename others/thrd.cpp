#include <iostream>
#include <thread>
using namespace std;

int x=0;

void f(int i) {
    cout << "Thread " << i << " is running" << endl;
    x+=10;
    cout << "Thread " << i << " is done" << endl;
}


int main() {
    int nr = 5;
    thread t[nr];
    for(int i = 0; i < nr; ++i) {
        t[i] = thread(f, i);
    }

    cout << "Main thread is running" << endl;

    for(int i = 0; i < nr; ++i) {
        t[i].join();
    }
    cout << "X = " << x << endl;

    cout << "All threads are done" << endl;
    return 0;
}