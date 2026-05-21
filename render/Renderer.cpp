#include "Renderer.h"
#include "../world/World.h"
#include "../world/Entity.h"
#include "../world/Player.h"
#include "../world/Camera.h"
#include "Mesh.h"
#include "Texture.h"

#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstring>

// ---------------------------------------------------------------------------
//  PBR Shader sources
// ---------------------------------------------------------------------------
static const char* pbrVertSrc = R"(
#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aColor;

out vec3 WorldPos;
out vec3 WorldNormal;
out vec2 TexCoord;
out vec4 Color;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform mat4 uNormalMat;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    WorldPos = worldPos.xyz;
    gl_Position = uViewProj * worldPos;
    WorldNormal = normalize(mat3(uNormalMat) * aNormal);
    TexCoord = aTexCoord;
    Color = aColor;
}
)";

static const char* pbrFragSrc = R"(
#version 460 core
in vec3 WorldPos;
in vec3 WorldNormal;
in vec2 TexCoord;
in vec4 Color;

out vec4 FragColor;

uniform vec3 uCamPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform float uAmbient;
uniform sampler2D uAlbedoTex;
uniform bool uHasTexture;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (3.14159265 * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

void main() {
    vec3 albedo = uHasTexture ? texture(uAlbedoTex, TexCoord).rgb : Color.rgb;

    float metallic = 0.0;
    float roughness = 0.5;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 N = normalize(WorldNormal);
    vec3 V = normalize(uCamPos - WorldPos);
    vec3 L = normalize(-uLightDir);
    vec3 H = normalize(V + L);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    float NdotL = max(dot(N, L), 0.0);

    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0001) * max(dot(N, L), 0.0001) + 0.0001);
    vec3 diffuse = kD * albedo / 3.14159265;

    vec3 Lo = (diffuse + specular) * uLightColor * NdotL;
    vec3 ambient = uAmbient * albedo;
    FragColor = vec4(ambient + Lo, 1.0);
}
)";

// ---------------------------------------------------------------------------
//  Shader compilation helpers
// ---------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!vert || !frag) return 0;

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader link error: " << infoLog << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

namespace abel {

// ---------------------------------------------------------------------------
//  GPU buffer for a mesh
// ---------------------------------------------------------------------------
struct MeshGPU {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    int indexCount = 0;
    bool hasTexture = false;
    GLuint textureID = 0;
};

// ---------------------------------------------------------------------------
//  Renderer
// ---------------------------------------------------------------------------
Renderer::Renderer()
    : window_(nullptr), pbrProgram_(0)
{}

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::init() {
    // GLFW init
    if (!glfwInit()) {
        std::cerr << "GLFW init failed" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(1280, 720, "Abel", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Window creation failed" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    // No GLAD – we use glcorearb.h + GLFW

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glClearColor(0.1f, 0.15f, 0.25f, 1.0f);

    // Compile PBR shader
    pbrProgram_ = createProgram(pbrVertSrc, pbrFragSrc);
    if (!pbrProgram_) return false;

    // Retrieve uniform locations
    glUseProgram(pbrProgram_);
    loc_uModel_      = glGetUniformLocation(pbrProgram_, "uModel");
    loc_uViewProj_   = glGetUniformLocation(pbrProgram_, "uViewProj");
    loc_uNormalMat_  = glGetUniformLocation(pbrProgram_, "uNormalMat");
    loc_uCamPos_     = glGetUniformLocation(pbrProgram_, "uCamPos");
    loc_uLightDir_   = glGetUniformLocation(pbrProgram_, "uLightDir");
    loc_uLightColor_ = glGetUniformLocation(pbrProgram_, "uLightColor");
    loc_uAmbient_    = glGetUniformLocation(pbrProgram_, "uAmbient");
    loc_uAlbedoTex_  = glGetUniformLocation(pbrProgram_, "uAlbedoTex");
    loc_uHasTexture_ = glGetUniformLocation(pbrProgram_, "uHasTexture");
    glUseProgram(0);

    return true;
}
void Renderer::shutdown() {
    // Delete all cached GPU buffers
    for (auto& [meshPtr, gpu] : gpuCache_) {
        glDeleteVertexArrays(1, &gpu.vao);
        glDeleteBuffers(1, &gpu.vbo);
        glDeleteBuffers(1, &gpu.ebo);
        if (gpu.textureID) glDeleteTextures(1, &gpu.textureID);
    }
    gpuCache_.clear();

    if (pbrProgram_) glDeleteProgram(pbrProgram_);
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

void Renderer::draw(const World& world, const Camera& camera) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!pbrProgram_ || !window_) return;

    glm::mat4 viewProj = camera.getProjection() * camera.getView();

    glUseProgram(pbrProgram_);
    // Global lighting
    glUniform3fv(loc_uLightDir_, 1, glm::value_ptr(lightDir_));
    glUniform3fv(loc_uLightColor_, 1, glm::value_ptr(lightColor_));
    glUniform1f(loc_uAmbient_, ambient_);

    for (const auto& entityPtr : world.getAllEntities()) {
        const Entity* entity = entityPtr.get();
        if (!entity || !entity->isVisible()) continue;

        const Mesh* mesh = entity->getMesh();
        if (!mesh) continue;

        MeshGPU& gpu = getOrCreateGPU(*mesh);

        glm::mat4 model = entity->getTransform();
        glm::mat4 normalMat = glm::transpose(glm::inverse(model));

        glUniformMatrix4fv(loc_uModel_, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(loc_uViewProj_, 1, GL_FALSE, glm::value_ptr(viewProj));
        glUniformMatrix4fv(loc_uNormalMat_, 1, GL_FALSE, glm::value_ptr(normalMat));
        glUniform3fv(loc_uCamPos_, 1, glm::value_ptr(camera.getPosition()));

        if (gpu.hasTexture) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gpu.textureID);
            glUniform1i(loc_uAlbedoTex_, 0);
            glUniform1i(loc_uHasTexture_, GL_TRUE);
        } else {
            glUniform1i(loc_uHasTexture_, GL_FALSE);
        }

        glBindVertexArray(gpu.vao);
        glDrawElements(GL_TRIANGLES, gpu.indexCount, GL_UNSIGNED_INT, nullptr);
    }

    // Optionally draw the player
    const Player* player = world.getPlayer();
    if (player) {
        const Mesh* playerMesh = player->getMesh();
        if (playerMesh) {
            MeshGPU& gpu = getOrCreateGPU(*playerMesh);

            glm::mat4 model = player->getTransform();
            glm::mat4 normalMat = glm::transpose(glm::inverse(model));

            glUniformMatrix4fv(loc_uModel_, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(loc_uViewProj_, 1, GL_FALSE, glm::value_ptr(viewProj));
            glUniformMatrix4fv(loc_uNormalMat_, 1, GL_FALSE, glm::value_ptr(normalMat));
            glUniform3fv(loc_uCamPos_, 1, glm::value_ptr(camera.getPosition()));

            if (gpu.hasTexture) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, gpu.textureID);
                glUniform1i(loc_uAlbedoTex_, 0);
                glUniform1i(loc_uHasTexture_, GL_TRUE);
            } else {
                glUniform1i(loc_uHasTexture_, GL_FALSE);
            }

            glBindVertexArray(gpu.vao);
            glDrawElements(GL_TRIANGLES, gpu.indexCount, GL_UNSIGNED_INT, nullptr);
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

MeshGPU& Renderer::getOrCreateGPU(const Mesh& mesh) {
    auto it = gpuCache_.find(&mesh);
    if (it != gpuCache_.end()) return it->second;

    MeshGPU gpu;
    gpu.indexCount = static_cast<int>(mesh.getIndices().size());
    gpu.hasTexture = false;
    gpu.textureID = 0;

    // Create VAO, VBO, EBO
    glGenVertexArrays(1, &gpu.vao);
    glGenBuffers(1, &gpu.vbo);
    glGenBuffers(1, &gpu.ebo);

    glBindVertexArray(gpu.vao);
    glBindBuffer(GL_ARRAY_BUFFER, gpu.vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.getVertices().size() * sizeof(Vertex),
                 mesh.getVertices().data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.getIndices().size() * sizeof(unsigned int),
                 mesh.getIndices().data(), GL_STATIC_DRAW);

    // Vertex layout
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nx));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, u));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, r));
    glEnableVertexAttribArray(3);

    // Check for texture
    const Texture* tex = mesh.getTexture();
    if (tex && tex->isValid()) {
        glGenTextures(1, &gpu.textureID);
        glBindTexture(GL_TEXTURE_2D, gpu.textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex->width(), tex->height(), 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, tex->data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tex->hasMipmaps() ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        gpu.hasTexture = true;
    }

    glBindVertexArray(0);

    auto [inserted, _] = gpuCache_.emplace(&mesh, std::move(gpu));
    return inserted->second;
}

} // namespace abel
