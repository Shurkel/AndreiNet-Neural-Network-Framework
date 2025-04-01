#include "../demos/demo2.cpp"

//0.337018s
//0.35 - 36 

using namespace std;

int main()
{
    //ofstream l;
    cls
    string outputfile = "logs.out";
    //setLogFile(outputfile);
    
    cout << MAGENTA << "\n[*]WILKOMMEN ZUM" << BOLDCYAN << " ANDREINET"
    << MAGENTA << "\n[+]Letztes Update:" <<BOLDWHITE << " 04.08.2024"
    << MAGENTA << "\n[+]Version:" << BOLDWHITE << " 1.0.1"
    << MAGENTA << "\n[+]Siehe Github fur mehrere Informationen:" << BOLDWHITE << " https://github.com/Shurkel/PrjSclNN"
    << YELLOW << "\n[ENTER]" << RESET << " um fortzufahren";
    cin.get();
    cls
    runDemo(outputfile); 

    
}

//one
//0.099503s
//0.099505s
//0.100508s
//two
//0.113506s
//0.111506s
//0.113263s