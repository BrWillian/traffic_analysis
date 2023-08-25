#include "../include/tracker.h"

Tracker::Tracker()
        : nextObjectID(0), maxDisappeared(1){
}

double Tracker::calcIoU(const std::vector<float> &bbox1, const std::vector<float> &bbox2) {
    float x1 = std::max(bbox1[0], bbox2[0]);
    float y1 = std::max(bbox1[1], bbox2[1]);
    float x2 = std::min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]);
    float y2 = std::min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3]);

    if (x2 <= x1 || y2 <= y1)
        return 0.0;

    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = bbox1[2] * bbox1[3];
    float area2 = bbox2[2] * bbox2[3];
    float unionArea = area1 + area2 - intersection;

    return intersection / unionArea;
}

void Tracker::register_Object(const std::vector<float> &bbox) {
    int objID = nextObjectID++;
    objects.emplace_back(objID, bbox);
}

void Tracker::deleteObject(int objectID) {
    objects.erase(std::remove_if(objects.begin(), objects.end(), [objectID](const auto &elem) {
        return elem.first == objectID;
    }), objects.end());
}
void Tracker::update(std::vector<Vehicle::Detection>& detections)
{
    if (detections.empty()) {
        for (auto& item : disappeared)
            item.second++;

        auto it = std::remove_if(objects.begin(), objects.end(), [this](const auto& elem) {
            int objectID = elem.first;
            return disappeared[objectID] > maxDisappeared;
        });
        objects.erase(it, objects.end());

        for (auto it = disappeared.begin(); it != disappeared.end();) {
            if (it->second > maxDisappeared)
                it = disappeared.erase(it);
            else
                ++it;
        }

        return;
    }if (objects.empty()) {
        for (auto& detection : detections) {
            std::vector<float> bbox(detection.bbox, detection.bbox + 4);
            detection.id = nextObjectID;
            register_Object(bbox);
        }
    } else {
        std::vector<int> objectIDs;
        std::vector<std::vector<float>> objectBboxes;
        for (const auto& object : objects) {
            objectIDs.push_back(object.first);
            objectBboxes.push_back(object.second);
        }

        std::vector<std::vector<float>> iouMatrix(objectBboxes.size(), std::vector<float>(detections.size()));
        for (size_t i = 0; i < objectBboxes.size(); ++i) {
            for (size_t j = 0; j < detections.size(); ++j) {
                std::vector<float> bbox(detections[j].bbox, detections[j].bbox + 4);
                iouMatrix[i][j] = calcIoU(objectBboxes[i], bbox);
            }
        }

        std::set<int> usedRows;
        std::set<int> usedCols;

        while (true) {
            double maxIoU = 0.0;
            size_t maxRow = 0;
            size_t maxCol = 0;

            for (size_t i = 0; i < iouMatrix.size(); ++i) {
                for (size_t j = 0; j < iouMatrix[i].size(); ++j) {
                    if (iouMatrix[i][j] > maxIoU) {
                        maxIoU = iouMatrix[i][j];
                        maxRow = i;
                        maxCol = j;
                    }
                }
            }

            if (maxIoU < 0.25)
                break;

            if (usedRows.count(maxRow) || usedCols.count(maxCol)) {
                iouMatrix[maxRow][maxCol] = 0.0;
                continue;
            }

            int objectID = objectIDs[maxRow];
            std::vector<float> bbox(detections[maxCol].bbox, detections[maxCol].bbox + 4);
            detections[maxCol].id = objectID;
            objects[maxRow].second = bbox;
            disappeared[objectID] = 0;

            usedRows.insert(maxRow);
            usedCols.insert(maxCol);

            iouMatrix[maxRow][maxCol] = 0.0;
        }

        std::set<int> objRows;
        std::set<int> inpCols;

        for (size_t i = 0; i < objectBboxes.size(); ++i)
            objRows.insert(i);
        for (size_t i = 0; i < detections.size(); ++i)
            inpCols.insert(i);

        std::set<int> unusedRows;
        std::set<int> unusedCols;

        std::set_difference(objRows.begin(), objRows.end(), usedRows.begin(), usedRows.end(),
                            std::inserter(unusedRows, unusedRows.begin()));
        std::set_difference(inpCols.begin(), inpCols.end(), usedCols.begin(), usedCols.end(),
                            std::inserter(unusedCols, unusedCols.begin()));

        for (const auto& row : unusedRows) {
            int objectID = objectIDs[row];
            disappeared[objectID] += 1;

            if (disappeared[objectID] > maxDisappeared)
                deleteObject(objectID);
        }

        for (const auto& col : unusedCols) {
            std::vector<float> bbox(detections[col].bbox, detections[col].bbox + 4);
            detections[col].id = nextObjectID;
            register_Object(bbox);
        }
    }
}