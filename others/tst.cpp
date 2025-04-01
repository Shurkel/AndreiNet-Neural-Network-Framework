
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;

/*
  * PRIETENE
int div(int n) {
    int s = 0;
    for (int i = 1; i <= n/2; i++)
        if (n % i == 0)
                s += i;
    return s;
}
int main() {
    int a, b;
    cin >> a >> b;
    if (div(a) == b && div(b) == a) {
        cout << "Prietene\n";
    } else {
        cout << "Nu prietene\n";
    }
    return 0;
}
*/

/*
  * 2965. Find Missing and Repeated Values
    9 1 7
    8 9 2
    3 4 6
int main()
{
    int n = 3;
    // cin >> n;
    int m[n][n] = {{9, 1, 7}, {8, 9, 2}, {3, 4, 6}};

    int a, b;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                for (int l = 0; l < n; l++)
                    if (m[i][j] == m[k][l] && (i != k || j != l))
                    {
                        a = m[i][j];
                        break;
                    }
    int v[n * n] = {0};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            v[m[i][j]] = 1;

    for (int i = 1; i <= n * n; i++)
        if (v[i] == 0)
        {
            b = i;
            break;
        }
    cout << a << ' ' << b;
}
*/
/*
 * Se dă un șir de n numere întregi. Determină perechea de numere care, înmulțite, dau produsul maxim posibil.

int main()
{
    int n;
    cin >> n;
    int v[n];
    for (int i = 0; i < n; i++)
        cin >> v[i];
    int a=0, b=0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            if(v[i]*v[j] > a*b && i != j)
            {
                a = v[i];
                b = v[j];
            }
    cout << a << ' ' << b;

}
*/

/*
    *Verificare Număr Armstrong Generalizat
    Enunț: Un număr Armstrong generalizat este un număr egal cu suma cifrelor 
    sale ridicate la puterea numărului de cifre. Să se verifice dacă un număr este Armstrong generalizat.


bool arm(int n)
{
    int a = n;
    int s = 0;
    int cnt = 0;
    while(a > 0)
    {
        cnt++;
        a /= 10;
    }
    a = n;
    while(a > 0)
    {
        s += pow(a % 10, cnt);
        a /= 10;
    }
    if (s == n)
        return true;
}

int main()
{
    int n;
    cin >> n;
    if(arm(n))
        cout << "DA";
    else
        cout << "NU";
}
*/


/*
* Copiere continut dintr-un fisier date.in in alt fisier date.text

int main()
{
    ifstream f("date.in");
    ofstream fout("date.txt");
    while(!f.eof())
    {
        string s;
        getline(f, s);
        fout << s << '\n';
    }

}
*/


/*
*Descompune un număr în factori primi și afișează numărul de divizori.


int main()
{
    int n;
    cin >> n;
    int div = 2, cnt = 0;
    while(n > 1)
    {
        
        while(n % div == 0)
        {
            n /= div;
            
        }
        cnt++;
        div++;
    }
    cout << cnt;
}
*/


/*
* #58 CMMDC
* Să se scrie un program care să determine cel mai mare divizor comun a două numere naturale citite de la tastatură.

int cmmdc(int a, int b)
{
    if(b == 0)
        return a;
    else if (a == 0)
        return b;
    return cmmdc(b, a % b);
}
int main()
{
    int a, b;
    cin >> a >> b;
    cout << cmmdc(a, b);
}
*/


/*
* Subprogramul produs are doi parametri, a și b, prin care primește câte un număr
* natural din intervalul [1,103]. Subprogramul returnează produsul divizorilor naturali
* comuni lui a și b. Scrieți definiția completă a subprogramului.  
Exemplu:  dacă  a=20  și  b=12,  atunci subprogramul returnează valoarea  8 
(1∙2∙4=8). 



int produs(int a, int b)
{
    int p = 1;
    for(int i = 1; i <= min(a,b); i++)
        if(a % i == 0 && b % i == 0)
            p *= i;
    return p;
}
int main()
{
    int a, b;
    cin >> a >> b;
    cout << produs(a, b);
}
*/

/*
* Pentru a studia un metal, s-a urmărit comportamentul său într-o succesiune de pași, la fiecare pas metalul fiind supus
*  unei anumite temperaturi. Pașii sunt numerotați cu valori naturale consecutive, începând de la 1. Un pas se numește
* reprezentativ dacă la niciunul dintre pașii anteriori nu este utilizată o temperatură strict mai mare decât la acest pas. 
* Dacă există o secvență de pași consecutivi la care se utilizează aceeași temperatură,
* se consideră reprezentativ doar primul pas din secvență.
* Fișierul bac.txt conține cel mult 10^6 numere naturale din intervalul [0,104], separate prin câte un spațiu,
* reprezentând temperaturile la care este supus metalul, în ordinea pașilor corespunzători. 
* Se cere să se afișeze pe ecran, separați prin câte un spațiu, pașii
* reprezentativi pentru datele din fișier. 
Exemplu: dacă fișierul conține numerele 7 4 9 10 10 10 3 9 2 10 10 8 2 30  se afișează pe ecran                  
1 3 4 10 14


7 4 9 10 10 10 3 9 2 10 10 8 2 30
n n n n
m m m
s x s 

int main()
{
    ifstream f("bac.txt");
    int n, maxi= INT16_MIN, cnt = 1;
    
    while(!f.eof())
    {
        f>>n;
        if(n > maxi)
        {
            cout << cnt << '\n';
            maxi = n;
        }
        cnt++;
    }
}
*/

/*

*Problemă: Se dă o scară cu N trepte. Un individ se află în partea de jos a scării și poate să urce câte o
* treaptă la un pas, sau câte două trepte la un pas. În câte moduri poate urca scara?
Exemplu: Observăm că dacă scara are o treaptă, ea poate fi urcată într-un singur mod, 
iar dacă are două trepte, sunt două modalități de a urca scara: doi pași de o treaptă 
sau un un pas de două trepte. Pentru N=4, scara poate fi urcată în 5 moduri:



int main()
{
    int n;
    cin >> n;



}
*/
/*

* Să se determine o secvență de elemente de lungime k cu suma elementelor maximă.
*/

int main()
{
    int n, k;
    cin >> n >> k;
    


    int v[n];
    for(int i = 0; i < n; i++) 
        cin >> v[i];
    




    return 0;
}






