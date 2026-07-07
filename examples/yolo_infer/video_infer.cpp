#include <iostream>
#include <string>
#include <thread>

#include "runner/detect/YoloDetector.h"
#include "visual/painter.h"
#include "queue/SPSCQueue.h"
#include "Timer.h"

using namespace std;

void display(SPSCQueue<cv::Mat>& imgs) {
    auto time_start = std::chrono::high_resolution_clock::now();
    size_t size = 0;
    while (true) {
        cv::Mat img;
        bool success = imgs.pop(img);
        if (success) ++size;

        auto time_end = std::chrono::high_resolution_clock::now();
        size_t spend = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

        if (spend > 1000) {
            std::cout << "FPS: " << size * 1000 / spend << std::endl;
            time_start = time_end;
            size = 0;
        }
    }
}

int main(int argc, char const *argv[])
{
    
    const std::string& video_path = "assets/video.mp4";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera!" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::string config_path = "config/model_config.yaml";
    std::unique_ptr<Detector> detector = Detector::Load<YoloDetector>(config_path);

    SPSCQueue<cv::Mat> imgs(100);

    std::thread th(display, std::ref(imgs));


    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // TIMER_START();
        std::vector<Detection> dets = detector->detect(frame);
        // TIMER_FINISH();
        draw_boxs(frame, dets);
        imgs.push(std::move(frame));
    }

    th.join();
    cv::waitKey(0);

    return 0;
}
