#if(WITH_GPU AND TENSORRT_FOUND)
#  inference_analysis_test(video_tester SRCS video_tester.cc
#     EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps} ir_pass_manager analysis_predictor SERIAL)
  inference_analysis_test(faster_rcnn_tester SRCS faster_rcnn_tester.cc
     EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps} ir_pass_manager analysis_predictor SERIAL)
  inference_analysis_test(ocr_plate_tester SRCS ocr_plate_tester.cc
     EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps} ir_pass_manager analysis_predictor SERIAL)
  inference_analysis_test(machine_reading_comprehension_tester SRCS machine_reading_comprehension_tester.cc
     EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps} ir_pass_manager analysis_predictor SERIAL)
#endif()
