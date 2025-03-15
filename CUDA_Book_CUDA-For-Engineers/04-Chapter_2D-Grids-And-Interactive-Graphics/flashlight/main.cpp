#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "interactions.h"

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
cudaGraphicsResource *cuda_pbo_resource;

void checkGLError(const char *label)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        printf("OpenGL error at %s: %s\n", label, gluErrorString(err));
    }
}
void printOpenGLInfo() {
    const GLubyte* renderer = glGetString(GL_RENDERER);    // Renderer string
    const GLubyte* vendor = glGetString(GL_VENDOR);        // Vendor string
    const GLubyte* version = glGetString(GL_VERSION);      // OpenGL version string
    const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);  // GLSL version string
    
    printf("Renderer: %s\n", renderer);
    printf("Vendor: %s\n", vendor);
    printf("OpenGL Version: %s\n", version);
    printf("GLSL Version: %s\n", glslVersion);
}

void render()
{
    uchar4 *d_out = 0;
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
    {
        printf("Error mapping resources: %s\n", cudaGetErrorString(err));
    }

    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
    kernelLauncher(d_out, W, H, loc);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    checkGLError("glBindBuffer - PBO");

    glBindTexture(GL_TEXTURE_2D, tex);
    checkGLError("glBindTexture - Texture");

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    checkGLError("glTexImage2D");

    glEnable(GL_TEXTURE_2D); // Enable 2D texture rendering
    checkGLError("glEnable - GL_TEXTURE_2D");

    // Draw a quad with texture coordinates
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, H);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(W, H);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(W, 0);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    checkGLError("glDisable - GL_TEXTURE_2D");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    render();
    drawTexture();
    glutSwapBuffers();
}

void initGLUT(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(W, H);
    glutCreateWindow(TITLE_STRING);

#ifndef __APPLE__
    if (glewInit() != GLEW_OK)
    {
        printf("GLEW initialization failed!\n");
        exit(1);
    }
    else
    {
        printf("GLEW initialized\n");
    }
#endif

    glViewport(0, 0, W, H); // Set the viewport
}

void initPixelBuffer()
{
    glGenBuffers(1, &pbo);
    checkGLError("glGenBuffers - PBO");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    checkGLError("glBindBuffer - Verify Binding");

    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * 600 * 600 * sizeof(GLubyte), 0, GL_STREAM_DRAW);
    checkGLError("glBufferData - GL_DYNAMIC_DRAW");
    // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); 

    // Verify PBO buffer size
    GLint bufferSize = 0;
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bufferSize);
    if (bufferSize == 0)
    {
        printf("Error: OpenGL buffer creation failed!\n");
    }
    else
    {
        printf("PBO buffer size: %d\n", bufferSize);
    }

    glGenTextures(1, &tex);
    checkGLError("glGenTextures - Texture");

    glBindTexture(GL_TEXTURE_2D, tex);
    checkGLError("glBindTexture - Texture");

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    checkGLError("glBindBuffer - before registering");

    // Verify that the PBO is correctly bound
    GLint currentPBO;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &currentPBO);
    if (currentPBO != pbo)
    {
        printf("Error: PBO is not bound correctly before registration.\n");
    }
    else
    {
        printf("PBO is bound correctly: %d\n", currentPBO);
    }

    cudaError_t contextError = cudaFree(0); // Initialize CUDA context
    if (contextError != cudaSuccess)
    {
        printf("CUDA context initialization error: %s\n", cudaGetErrorString(contextError));
    }
    else
    {
        printf("CUDA context initialized\n");
    }
    cudaGraphicsResource *cuda_pbo_resource;

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        printf("Error registering buffer with WriteDiscard flag: %s\n", cudaGetErrorString(err));

        // Try registering without the flags for debugging
        err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsNone);
        if (err != cudaSuccess)
        {
            printf("Error registering buffer with no flags: %s\n", cudaGetErrorString(err));
        }
    }
}

void exitfunc()
{
    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

int main(int argc, char **argv)
{
    printInstructions();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.canMapHostMemory)
    {
        printf("CUDA-OpenGL interop is supported.\n");
    }
    else
    {
        printf("CUDA-OpenGL interop is not supported.\n");
    }
    // int *dev;
    // cudaGetDeviceCount(dev);
    // printf("No of gpus: %d\n",*dev);
    // cudaSetDevice(0); // or whichever GPU OpenGL is using
    cudaSetDevice(0);  // If GPU 0 is where OpenGL context is running


    initGLUT(&argc, argv);
    printOpenGLInfo();

    gluOrtho2D(0, W, H, 0);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(handleSpecialKeypress);
    glutPassiveMotionFunc(mouseMove);
    glutMotionFunc(mouseDrag);
    glutDisplayFunc(display);

    initPixelBuffer();
    glutMainLoop();
    atexit(exitfunc);

    return 0;
}
