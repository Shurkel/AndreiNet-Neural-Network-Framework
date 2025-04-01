#include <iostream>
#include <cmath>
using namespace std;

double function(double x)
{
    return 0.2 * pow(x, 4) + 0.1 + pow(x, 3) - pow(x, 2) + 2;
}
double inputValue = 0.5;

void Learn(double learnRate)
{
    double h = 0.0001;
    double deltaOutput = function(inputValue+ h) - function(inputValue);
    double slope = deltaOutput / h;

    inputValue -= slope * learnRate;
}

int main()
{
    for (int i = 0; i < 100; i++)
    {
        Learn(0.07);
        cout << inputValue << endl;
    }
    return 0;
}