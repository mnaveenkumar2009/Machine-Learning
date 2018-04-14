#include <bits/stdc++.h>
#define f(i,n) for(i=0;i<n;i++)
#define ff first
#define ss second
#define ld long double
#define ll long long
using namespace std;
ld a=0,b=0,alpha=0.01;
ll n;
vector <ld> data_X,data_Y;
ld h(ld x){
    return a*x+b;
}
ld cost(){
    ld tcost=0;
    ll i;
    f(i,n){
        tcost+=(h(data_X[i])-data_Y[i])*(h(data_X[i])-data_Y[i]);
    }
    tcost/=2*n;
    return tcost;
}
void gradient_descent(){
    ld bdec=0,adec=0;
    ll i;
    f(i,n){
        bdec+=h(data_X[i])-data_Y[i];
        adec+=(h(data_X[i])-data_Y[i])*(data_X[i]);
    }

    a-=adec*alpha;
    b-=bdec*alpha;
}
int main(){
    /*ios_base::sync_with_stdio(false);
    cin.tie(NULL);*/
    ll i;
    cout<<"Enter number of data samples";
    cin>>n;
    data_X.resize(n);
    data_Y.resize(n);
    cout<<"Enter the data X";
    f(i,n){
        cin>>data_X[i];
    }
    cout<<"Enter the data Y";
    f(i,n){
        cin>>data_Y[i];
    }

    f(i,1000){
        gradient_descent();
        if(i%100==0)
        cout<<a<<" "<<b<<endl;
    }
    cout<<a<<" "<<b<<endl;
    return 0;
}

/*
Sample:

Enter number of data samples5
Enter the data X1 2 3 4 5
Enter the data Y3 4 5 6 7
0.85 0.25
1.18807 1.32102
1.08048 1.70946
1.03444 1.87567
1.01474 1.9468
1.00631 1.97723
1.0027 1.99026
1.00115 1.99583
1.00049 1.99822
1.00021 1.99924
1.00009 1.99967


*/