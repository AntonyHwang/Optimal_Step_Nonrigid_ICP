# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug

# Include any dependencies generated for this target.
include 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/depend.make

# Include the progress variables for this target.
include 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/progress.make

# Include the compile flags for this target's objects.
include 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/flags.make

# Object files for target nanogui
nanogui_OBJECTS =

# External object files for target nanogui
nanogui_EXTERNAL_OBJECTS = \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/ext/nanovg/src/nanovg.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/darwin.mm.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/nanogui_resources.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/glutil.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/common.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/widget.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/theme.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/layout.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/screen.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/label.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/window.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/popup.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/checkbox.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/button.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/popupbutton.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/combobox.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/progressbar.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/slider.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/messagedialog.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/textbox.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/imagepanel.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/imageview.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/vscrollpanel.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/colorwheel.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/colorpicker.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/graph.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/stackedwidget.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/tabheader.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/tabwidget.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/glcanvas.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/serializer.cpp.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/context.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/init.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/input.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/monitor.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/vulkan.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/window.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_init.m.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_joystick.m.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_monitor.m.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_window.m.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_time.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/posix_tls.c.o" \
"/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/nsgl_context.m.o"

3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/ext/nanovg/src/nanovg.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/darwin.mm.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/nanogui_resources.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/glutil.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/common.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/widget.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/theme.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/layout.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/screen.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/label.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/window.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/popup.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/checkbox.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/button.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/popupbutton.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/combobox.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/progressbar.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/slider.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/messagedialog.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/textbox.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/imagepanel.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/imageview.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/vscrollpanel.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/colorwheel.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/colorpicker.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/graph.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/stackedwidget.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/tabheader.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/tabwidget.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/glcanvas.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui-obj.dir/src/serializer.cpp.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/context.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/init.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/input.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/monitor.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/vulkan.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/window.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_init.m.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_joystick.m.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_monitor.m.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_window.m.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/cocoa_time.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/posix_tls.c.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/ext_build/glfw/src/CMakeFiles/glfw_objects.dir/nsgl_context.m.o
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/build.make
3rdparty/libigl/external/nanogui/libnanogui.dylib: 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX shared library libnanogui.dylib"
	cd /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nanogui.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/build: 3rdparty/libigl/external/nanogui/libnanogui.dylib

.PHONY : 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/build

3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/requires:

.PHONY : 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/requires

3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/clean:
	cd /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui && $(CMAKE_COMMAND) -P CMakeFiles/nanogui.dir/cmake_clean.cmake
.PHONY : 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/clean

3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/depend:
	cd /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/3rdparty/libigl/external/nanogui /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui /Users/antonyhwang/Desktop/Optimal_Step_Nonrigid_ICP/IGLFramework/cmake-build-debug/3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rdparty/libigl/external/nanogui/CMakeFiles/nanogui.dir/depend

