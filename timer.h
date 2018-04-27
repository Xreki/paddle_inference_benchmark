/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <time.h>
#include <iostream>

class Timer {
 public:
  Timer(std::string name) : name_(name) {
    clock_gettime(CLOCK_MONOTONIC, &tp_start);
  }
  ~Timer() {
    clock_gettime(CLOCK_MONOTONIC, &tp_end);
    float time = ((tp_end.tv_nsec - tp_start.tv_nsec) / 1000000.0f) +
                 (tp_end.tv_sec - tp_start.tv_sec) * 1000;
    std::cout << "Time of " << name_ << ": " << time << " ms." << std::endl;
  }

 private:
  std::string name_;
  struct timespec tp_start;
  struct timespec tp_end;
};
