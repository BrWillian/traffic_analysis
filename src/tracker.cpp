#include "../include/tracker.h"

using namespace std;

Tracker::Tracker(int maxDisappeared)
{
    this->nextObjectID = 0;
    this->maxDisappeared = maxDisappeared;
}
double Tracker::calcDistance(double x1, double y1, double x2, double y2){
    double x = x1 - x2;
    double y = y1 - y2;
    double dist = std::sqrt((x * x) + (y * y));

    return dist;
}
void Tracker::register_Object(int cX, int cY){
    int objID = this->nextObjectID;
    this->objects.push_back({objID, {cX, cY}});
    this->disappeared.insert({objID, 0});
    this->nextObjectID += 1;
}
void Tracker::deleteObject(int objectID){
    this->objects.erase(remove_if(this->objects.begin(), this->objects.end(), [objectID](auto &elem) {
        return elem.first == objectID;
    }), this->objects.end());

    this->disappeared.erase(objectID);
}
std::vector<float>::size_type findMin(const std::vector<float> &v, std::vector<float>::size_type pos = 0){
    if (v.size() <= pos) return (v.size());
    std::vector<float>::size_type min = pos;
    for(std::vector<float>::size_type i = pos + 1; i< v.size(); i++){
        if(v[i] < v[min]) min = i;
    }
    return (min);
}
std::vector<std::pair<int, std::pair<int, int>>> Tracker::update(std::vector<Yolo::Detection> &dets) {
    if (dets.empty()) {
        auto it = this->disappeared.begin();
        while (it != this->disappeared.end()) {
            it->second++;
            if (it->second > this->maxDisappeared) {
                this->objects.erase(remove_if(this->objects.begin(), this->objects.end(), [it](auto &elem) {
                    return elem.first == it->first;
                }), this->objects.end());

                it = this->disappeared.erase(it);
            } else {
                ++it;
            }
        }
        return this->objects;
    }

    vector<pair<int, int>> inputCentroids;
    for (auto det : dets) {
        int cX = int(det.bbox[0] + det.bbox[2] / 2.0);
        int cY = int(det.bbox[1] + det.bbox[3] / 2.0);
        inputCentroids.push_back(make_pair(cX, cY));
    }

    if (this->objects.empty()) {
        for (auto i: inputCentroids) {
            this->register_Object(i.first, i.second);
        }
    }

    else {
        vector<int> objectIDs;
        vector<pair<int, int>> objectCentroids;
        for (auto object : this->objects) {
            objectIDs.push_back(object.first);
            objectCentroids.push_back(make_pair(object.second.first, object.second.second));
        }

        vector<vector<float>> Distances;
        for (size_t i = 0; i < objectCentroids.size(); ++i) {
            vector<float> temp_D;
            for (vector<vector<int>>::size_type j = 0; j < inputCentroids.size(); ++j) {
                double dist = calcDistance(objectCentroids[i].first, objectCentroids[i].second, inputCentroids[j].first,
                                           inputCentroids[j].second);

                temp_D.push_back(dist);
            }
            Distances.push_back(temp_D);
        }

        vector<int> cols;
        vector<int> rows;

        for (auto v: Distances) {
            auto temp = findMin(v);
            cols.push_back(temp);
        }

        vector<vector<float>> D_copy;
        for (auto v: Distances) {
            sort(v.begin(), v.end());
            D_copy.push_back(v);
        }

        vector<pair<float, int>> temp_rows;
        int k = 0;
        for (auto i: D_copy) {
            temp_rows.push_back(make_pair(i[0], k));
            k++;
        }

        for (auto const &x : temp_rows) {
            rows.push_back(x.second);
        }

        set<int> usedRows;
        set<int> usedCols;

        for (size_t i = 0; i < rows.size(); i++) {
            //if (usedRows.count(rows[i]) || usedCols.count(cols[i])) { continue; }
            int objectID = objectIDs[rows[i]];
            for (size_t t = 0; t < this->objects.size(); t++) {
                double dist = calcDistance(this->objects[t].second.first, this->objects[t].second.second, inputCentroids[cols[i]].first, inputCentroids[cols[i]].second);
                if(dist > 150) continue;

                if (this->objects[t].first == objectID) {
                    this->objects[t].second.first = inputCentroids[cols[i]].first;
                    this->objects[t].second.second = inputCentroids[cols[i]].second;
                }else{
                    deleteObject(objectID);
                }

            }
            this->disappeared[objectID] = 0;


            usedRows.insert(rows[i]);
            usedCols.insert(cols[i]);
        }

        set<int> objRows;
        set<int> inpCols;

        for (size_t i = 0; i < objectCentroids.size(); i++) {
            objRows.insert(i);
        }
        for (size_t i = 0; i < inputCentroids.size(); i++) {
            inpCols.insert(i);
        }

        set<int> unusedRows;
        set<int> unusedCols;

        set_difference(objRows.begin(), objRows.end(), usedRows.begin(), usedRows.end(),
                       inserter(unusedRows, unusedRows.begin()));
        set_difference(inpCols.begin(), inpCols.end(), usedCols.begin(), usedCols.end(),
                       inserter(unusedCols, unusedCols.begin()));


        if (objectCentroids.size() >= inputCentroids.size()) {
            for (auto row: unusedRows) {
                int objectID = objectIDs[row];
                this->disappeared[objectID] += 1;

                if (this->disappeared[objectID] > this->maxDisappeared) {
                    this->objects.erase(remove_if(this->objects.begin(), this->objects.end(), [objectID](auto &elem) {
                        return elem.first == objectID;
                    }), this->objects.end());

                    this->disappeared.erase(objectID);
                }
            }
        } else {
            for (auto col: unusedCols) {
                this->register_Object(inputCentroids[col].first, inputCentroids[col].second);
            }
        }
    }


    return this->objects;
}
