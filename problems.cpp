#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>
#include <iomanip>
#include <set>
#include <iterator>
#include <sstream>
#include <fstream>
#include <string_view>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <deque>

using namespace std;

//#include "test_runner.h"

vector<string_view> SplitIntoWords(string_view str) {
	vector<string_view> result;

	string compare = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";

	size_t pos = 0;
	const size_t pos_end = str.npos;
	while(true) {
		size_t space = str.find_first_not_of(compare, pos);
		result.push_back(
				space == pos_end
				? str.substr(pos)
				: str.substr(pos, space - pos));

		if(space == pos_end) {
			break;
		} else {
			pos = space + 1;
		}
	}
	return result;
}


int main() {
//	fstream input("022.txt");
	size_t n;
	cin >> n;
	bool up_or_low = false, num_start = false;
	string temps;
	cin >> temps;
	temps == "yes" ? up_or_low = true : up_or_low = false;
	cin >> temps;
	temps == "yes" ? num_start = true : num_start = false;
	unordered_set<string> keypool;
	for(size_t i = 0; i < n; i++) {
		cin >> temps;
		if(!up_or_low) {
			transform(temps.begin(), temps.end(), temps.begin(), [](unsigned char c) {return tolower(c);});
		}
		keypool.insert(temps);
	}

	set<string> unique_words;
	unordered_map<string, pair<size_t, size_t>> storage;
	temps.clear();
	while(getline(cin, temps)) {
		if(!up_or_low) {
			transform(temps.begin(), temps.end(), temps.begin(), [](unsigned char c) {return tolower(c);});
		}
		for(const auto& item : SplitIntoWords(temps)) {
			string word = string(item);
			if(!num_start) {
				if(word[0] >= '0' && word[0] <= '9') {
					continue;
				}
			} else if(num_start) {
				if(word[0] >= '0' && word[0] <= '9' && word.size() == 1) {
					continue;
				}
			}
			if(word != "" && keypool.count(word) == 0) {
				if(unique_words.count(word) == 0) {
					unique_words.insert(word);
					storage[word] = {1, unique_words.size()};
				} else if (unique_words.count(word) > 0) {
					storage[word].first += 1;
				}
			}
		}
	}
	size_t minpos = unique_words.size();
	size_t max_num = 0;
	vector<string> finalwords;
	for(const auto& [word, value] : storage) { //максимальное значение
		if(max_num <= value.first) {
			max_num = value.first;
		}
	}
	string result_word;
	for(const auto& [word, value] : storage) {
		if(value.first == max_num && minpos >= value.second) { //первое появление слова
			minpos = value.second;
			result_word = word;
		}
	}
//	cout << "keypool: " << keypool << endl;
//	cout << "unique_words: " << unique_words << endl;
//	cout << "storage: " << storage << endl;
	cout << result_word;

	return 0;
}
