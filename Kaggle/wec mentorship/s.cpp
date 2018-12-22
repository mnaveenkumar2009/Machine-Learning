#include <bits/stdc++.h>
using namespace std;

int main(){
    ifstream train("data/train.csv");
    string s;
    train>>s;
    map <int,int> xxx;
    while(train >> s){
        int i;
        for( i=s.length()-1;i>=0;i--)
            if(s[i]==',')
                break;        
        s=s.substr(i+1,s.length()-i-1);
        if(s.size()==1)
            s="0"+s;
        s[1]-='0';
        s[0]-='0';
        xxx[s[1]+10*s[0]]++;
    }
    vector <int> nums;
    vector <int> freq(35,0);
    for(auto it:xxx){
        cout<<it.first<<' '<<it.second<<'\n';
        freq[it.first]+=it.second;
        while(it.second--)
            nums.push_back(it.first);
    }

    ofstream xyz("data/out.csv");
    xyz<<"Id,Expected\n";
    ifstream sams("data/sams.csv");
    int i=0;
    srand(time(0));
    while(sams>>i){
            int a1=nums[rand()%((int)nums.size())],a2=nums[rand()%((int)nums.size())],a3=nums[rand()%((int)nums.size())];
            if(freq[a3]>freq[a1])swap(a1,a3);
            if(freq[a1]<freq[a2])swap(a1,a2);
            xyz<<i<<','<<a1<<' '<<a3<<' '<<a2<<'\n';
    }
}

/*
00 30
01 11
02 1
03 1
04 1
05 3
06 7
07 14
08 5
09 3
10 21
11 20
12 7
13 39
14 16
15 2
16 3
17 1
18 16
19 31
21 6
22 27
23 116
24 2
25 24
26 24
27 50
28 74
*/