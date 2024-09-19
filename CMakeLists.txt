cmake_minimum_required(VERSION 2.8)

project(SeetaFaceTest)
set(CMAKE_BUILD_TYPE Debug)
set(SRC_ROOT_PATH ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_ROOT_PATH ${PROJECT_SOURCE_DIR}/include)
set(LIB_ROOT_PATH ${PROJECT_SOURCE_DIR}/libs)
include_directories(${LIB_ROOT_PATH})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${INCLUDE_ROOT_PATH}/seetaface2/FaceDetector/include)
include_directories(${INCLUDE_ROOT_PATH}/seetaface2/FaceLandmarker/include)
include_directories(${INCLUDE_ROOT_PATH}/seetaface2/FaceRecognizer/include)
include_directories(${INCLUDE_ROOT_PATH}/seetaface2/FaceTracker/include)
include_directories(${INCLUDE_ROOT_PATH}/seetaface2/QualityAssessor/include)
include_directories(${INCLUDE_ROOT_PATH}/seetaface2/SeetaNet/include)
include_directories(${INCLUDE_ROOT_PATH})
include_directories(${INCLUDE_ROOT_PATH}/libtiff)

find_package(Threads REQUIRED)

add_executable(seeta src/main.cpp
        include/Struct_cv.h)



# target_link_libraries(seeta PUBLIC
#     ${LIB_ROOT_PATH}/opencv/libopencv_core.a
#     ${LIB_ROOT_PATH}/opencv/libopencv_imgproc.a
#     ${LIB_ROOT_PATH}/opencv/libopencv_imgcodecs.a
#     ${LIB_ROOT_PATH}/opencv/3rdparty/liblibwebp.a  # liblibwebp.a 必须放在依赖它的库后面
#         ${LIB_ROOT_PATH}/opencv/3rdparty/liblibjpeg-turbo.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/liblibopenjp2.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/liblibpng.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/liblibprotobuf.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/liblibtiff.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/liblibwebp.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/libquirc.a
#         ${LIB_ROOT_PATH}/opencv/3rdparty/libzlib.a
#     )

    link_directories(${LIB_ROOT_PATH}/opencv/)
    link_directories(${LIB_ROOT_PATH}/opencv/3rdparty/)
    target_link_libraries(seeta
    ${LIB_ROOT_PATH}/opencv/libopencv_highgui.a ${LIB_ROOT_PATH}/opencv/libopencv_imgcodecs.a 
    ${LIB_ROOT_PATH}/opencv/libopencv_imgproc.a   ${LIB_ROOT_PATH}/opencv/libopencv_core.a
    ${LIB_ROOT_PATH}/opencv/3rdparty/libade.a   ${LIB_ROOT_PATH}/opencv/3rdparty/libIlmImf.a  ${LIB_ROOT_PATH}/opencv/3rdparty/liblibjpeg-turbo.a ${LIB_ROOT_PATH}/opencv/3rdparty/liblibopenjp2.a
    ${LIB_ROOT_PATH}/opencv/3rdparty/liblibpng.a ${LIB_ROOT_PATH}/opencv/3rdparty/liblibprotobuf.a 
    ${LIB_ROOT_PATH}/opencv/3rdparty/liblibtiff.a ${LIB_ROOT_PATH}/opencv/3rdparty/liblibwebp.a 
    ${LIB_ROOT_PATH}/opencv/3rdparty/libquirc.a ${LIB_ROOT_PATH}/opencv/3rdparty/libzlib.a
     z pthread dl)
    
    target_link_libraries(seeta
                        
                        ${LIB_ROOT_PATH}/seeta/libSeetaFaceDetector_d.a
                        ${LIB_ROOT_PATH}/seeta/libSeetaFaceLandmarker_d.a
                        ${LIB_ROOT_PATH}/seeta/libSeetaFaceRecognizer_d.a
                        ${LIB_ROOT_PATH}/seeta/libSeetaFaceTracker_d.a
                        ${LIB_ROOT_PATH}/seeta/libSeetaNet_d.a
                        ${LIB_ROOT_PATH}/seeta/libSeetaQualityAssessor_d.a
                        

                        ${CMAKE_THREAD_LIBS_INIT}
                        dl
                        m
                    )