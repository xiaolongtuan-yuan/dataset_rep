#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    // 获取家目录路径
    const char *homeDir = getenv("HOME");
    if (homeDir == nullptr) {
        std::cerr << "Failed to get home directory" << std::endl;
        return 1;
    }

    // 拼接文件夹路径
    std::ostringstream oss;
    oss << homeDir << "/test_directory_creation";
    std::string folder_path = oss.str();

    // 尝试创建目录
    struct stat info;
    if (stat(folder_path.c_str(), &info) != 0) {
        // 目录不存在，尝试创建
        if (mkdir(folder_path.c_str(), 0777) != 0) {
            std::cerr << "Failed to create directory: " << folder_path << std::endl;
            return 1;
        } else {
            std::cout << "Directory created successfully: " << folder_path << std::endl;
        }
    } else if (info.st_mode & S_IFDIR) {
        std::cout << "Directory already exists: " << folder_path << std::endl;
    } else {
        std::cerr << "Failed to create directory: " << folder_path << std::endl;
        return 1;
    }

    return 0;
}
