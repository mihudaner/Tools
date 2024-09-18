#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int comb(int n) {
    if (n < 2) return 0;
    return n * (n - 1) / 2;
}

int main() {
    int n;
    
    while (cin >> n) {
        vector<int> results;
        results.push_back(0);
        for (int i = n - 1; i <= comb(n); ++i) {
            results.push_back(i);
        }

        for (int i = 0; i < results.size(); ++i) {
            if (i > 0) cout << " ";
            cout << results[i];
        }
        cout << endl;
    }

    return 0;
}
