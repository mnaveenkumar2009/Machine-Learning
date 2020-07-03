#include <bits/stdc++.h>
using namespace std;
#define ld long double
#define ll long long

int main(){
    unordered_map <ll, ld> years[12];
    ll y, m, d;
    ld val;
    while(cin >> y >> m >> d >> val){
        years[m - 1][y] = max(years[m - 1][y], val);
    }
    for(y = 1980; y <= 2009; y++){
        ld mini = 100;
        for(int j = 0; j < 12; j++){
            mini = min(mini, years[j][y]);
        }
        for(int j = 0; j < 12; j++){
            // cout << years[j][y]/mini << ' ';
            cout << years[j][y] << ' ';
        }
        cout << '\n';
    }
}