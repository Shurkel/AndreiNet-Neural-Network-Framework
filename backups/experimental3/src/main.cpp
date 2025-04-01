#include "../utils/anet.h"
using namespace std;

int main()
{
    vector<node> nodesA;
    nodesA.push_back(node(1, 0.0, 0));
    nodesA.push_back(node(1, 0.0, 0));
    nodesA.push_back(node(1, 0.0, 0));
    vector<node> nodesB;
    nodesB.push_back(node(0, 0.0, 0));
    nodesB.push_back(node(0, 0.0, 0));



    for(int i = 0; i < nodesA.size(); i++)
    {
        nodesA[i].setNextNodes({{&nodesB[0], 1.0}, {&nodesB[1], 1.0}});
    }


    
    for(int i = 0; i < nodesA.size(); i++)
    {
        nodesA[i].passValue();
    }
    cout << nodesB[0].getValue() << endl;
    cout << nodesB[1].getValue() << endl;
}