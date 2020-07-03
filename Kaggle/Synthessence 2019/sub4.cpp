#include <bits/stdc++.h>
using namespace std;
#define ld long double
#define ll long long

int main(){
    unordered_map <ll, ld> years[12];
    ll y, m, d;
    ld val;
    ifstream X ("test.csv");
    string s;
    while(cin >> y >> m >> d >> val){
        if(X >> s){
            if(val > 24)
                val = 24;
            cout <<fixed << setprecision(9) << s << ',' << val << '\n';
        }
        else{
            break;
        }
    }
}