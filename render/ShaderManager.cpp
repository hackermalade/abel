#include "ShaderManager.h"
#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
#include <GLFW/glfw3.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace abel {

static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::string log = infoLog;
        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation error:\n" + log);
    }
    return shader;
}

static std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open shader file: " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

ShaderManager::ShaderManager() {}
ShaderManager::~ShaderManager() { clear(); }

void ShaderManager::clear() {
    for (auto& [name, prog] : programs_)
        glDeleteProgram(prog);
    programs_.clear();
}

GLuint ShaderManager::loadFromFiles(const std::string& name,
                                    const std::string& vertPath,
                                    const std::string& fragPath) {
    if (programs_.count(name)) return programs_[name];
    std::string vertSrc = readFile(vertPath);
    std::string fragSrc = readFile(fragPath);
    GLuint program = compileFromSource(vertSrc.c_str(), fragSrc.c_str());
    programs_[name] = program;
    return program;
}

GLuint ShaderManager::compileFromSource(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::string log = infoLog;
        glDeleteProgram(program);
        glDeleteShader(vert);
        glDeleteShader(frag);
        throw std::runtime_error("Shader link error:\n" + log);
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

GLuint ShaderManager::getProgram(const std::string& name) const {
    auto it = programs_.find(name);
    if (it != programs_.end()) return it->second;
    return 0;
}

} // namespace abel
