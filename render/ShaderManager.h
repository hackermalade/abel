#pragma once
#include <string>
#include <unordered_map>
#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
#include <GLFW/glfw3.h>

namespace abel {

class ShaderManager {
public:
    ShaderManager();
    ~ShaderManager();

    void clear();

    // Load shader from files, cache by name
    GLuint loadFromFiles(const std::string& name,
                         const std::string& vertPath,
                         const std::string& fragPath);

    // Compile shader directly from source strings (not cached)
    GLuint compileFromSource(const char* vertSrc, const char* fragSrc);

    // Get a cached program by name (returns 0 if not found)
    GLuint getProgram(const std::string& name) const;

private:
    std::unordered_map<std::string, GLuint> programs_;
};

} // namespace abel
