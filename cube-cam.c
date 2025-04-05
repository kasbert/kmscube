/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#define _GNU_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "esUtil.h"

static struct {
	struct egl egl;

	GLfloat aspect;
	const struct gbm *gbm;

	GLuint program, blit_program;
	/* uniform handles: */
	GLint modelviewmatrix, modelviewprojectionmatrix, normalmatrix;
	GLint texture, blit_texture;
	GLuint vbo;
	GLuint positionsoffset, texcoordsoffset, normalsoffset;
	GLuint tex;

	/* video decoder: */
	struct decoder *decoder;
	int filenames_count, idx;
	const char *filenames[32];

	EGLSyncKHR last_fence;
} gl;

static const struct egl *egl = &gl.egl;

static const GLfloat vVertices[] = {
		// front
		-1.0f, -1.0f, +1.0f,
		+1.0f, -1.0f, +1.0f,
		-1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		// back
		+1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		+1.0f, +1.0f, -1.0f,
		-1.0f, +1.0f, -1.0f,
		// right
		+1.0f, -1.0f, +1.0f,
		+1.0f, -1.0f, -1.0f,
		+1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, -1.0f,
		// left
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, +1.0f,
		-1.0f, +1.0f, -1.0f,
		-1.0f, +1.0f, +1.0f,
		// top
		-1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		-1.0f, +1.0f, -1.0f,
		+1.0f, +1.0f, -1.0f,
		// bottom
		-1.0f, -1.0f, -1.0f,
		+1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, +1.0f,
		+1.0f, -1.0f, +1.0f,
};

static const GLfloat vTexCoords[] = {
		//front
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//back
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//right
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//left
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//top
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//bottom
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
};

static const GLfloat vNormals[] = {
		// front
		+0.0f, +0.0f, +1.0f, // forward
		+0.0f, +0.0f, +1.0f, // forward
		+0.0f, +0.0f, +1.0f, // forward
		+0.0f, +0.0f, +1.0f, // forward
		// back
		+0.0f, +0.0f, -1.0f, // backward
		+0.0f, +0.0f, -1.0f, // backward
		+0.0f, +0.0f, -1.0f, // backward
		+0.0f, +0.0f, -1.0f, // backward
		// right
		+1.0f, +0.0f, +0.0f, // right
		+1.0f, +0.0f, +0.0f, // right
		+1.0f, +0.0f, +0.0f, // right
		+1.0f, +0.0f, +0.0f, // right
		// left
		-1.0f, +0.0f, +0.0f, // left
		-1.0f, +0.0f, +0.0f, // left
		-1.0f, +0.0f, +0.0f, // left
		-1.0f, +0.0f, +0.0f, // left
		// top
		+0.0f, +1.0f, +0.0f, // up
		+0.0f, +1.0f, +0.0f, // up
		+0.0f, +1.0f, +0.0f, // up
		+0.0f, +1.0f, +0.0f, // up
		// bottom
		+0.0f, -1.0f, +0.0f, // down
		+0.0f, -1.0f, +0.0f, // down
		+0.0f, -1.0f, +0.0f, // down
		+0.0f, -1.0f, +0.0f  // down
};

static const char *blit_vs =
		"attribute vec4 in_position;        \n"
		"attribute vec2 in_TexCoord;        \n"
		"                                   \n"
		"varying vec2 vTexCoord;            \n"
		"                                   \n"
		"void main()                        \n"
		"{                                  \n"
		"    gl_Position = in_position;     \n"
		"    vTexCoord = in_TexCoord;       \n"
		"}                                  \n";

static const char *blit_fs =
		"#extension GL_OES_EGL_image_external : enable\n"
		"precision mediump float;           \n"
		"                                   \n"
		"uniform samplerExternalOES uTex;   \n"
		"                                   \n"
		"varying vec2 vTexCoord;            \n"
		"                                   \n"
		"void main()                        \n"
		"{                                  \n"
		"    gl_FragColor = texture2D(uTex, vTexCoord);\n"
		"}                                  \n";

static const char *vertex_shader_source =
		"uniform mat4 modelviewMatrix;      \n"
		"uniform mat4 modelviewprojectionMatrix;\n"
		"uniform mat3 normalMatrix;         \n"
		"                                   \n"
		"attribute vec4 in_position;        \n"
		"attribute vec2 in_TexCoord;        \n"
		"attribute vec3 in_normal;          \n"
		"                                   \n"
		"vec4 lightSource = vec4(2.0, 2.0, 20.0, 0.0);\n"
		"                                   \n"
		"varying vec4 vVaryingColor;        \n"
		"varying vec2 vTexCoord;            \n"
		"                                   \n"
		"void main()                        \n"
		"{                                  \n"
		"    gl_Position = modelviewprojectionMatrix * in_position;\n"
		"    vec3 vEyeNormal = normalMatrix * in_normal;\n"
		"    vec4 vPosition4 = modelviewMatrix * in_position;\n"
		"    vec3 vPosition3 = vPosition4.xyz / vPosition4.w;\n"
		"    vec3 vLightDir = normalize(lightSource.xyz - vPosition3);\n"
		"    float diff = max(0.0, dot(vEyeNormal, vLightDir));\n"
		"    vVaryingColor = vec4(diff * vec3(1.0, 1.0, 1.0), 1.0);\n"
		"    vTexCoord = in_TexCoord; \n"
		"}                            \n";

static const char *fragment_shader_source =
		"#extension GL_OES_EGL_image_external : enable\n"
		"precision mediump float;           \n"
		"                                   \n"
		"uniform samplerExternalOES uTex;   \n"
		"                                   \n"
		"varying vec4 vVaryingColor;        \n"
		"varying vec2 vTexCoord;            \n"
		"                                   \n"
		"void main()                        \n"
		"{                                  \n"
		"    gl_FragColor = vVaryingColor * texture2D(uTex, vTexCoord);\n"
		"}                                  \n";


#define CAM_BUFFERS 1
#define CAM_WIDTH 720
#define CAM_HEIGHT 480

#define USE_RGB 1
#ifdef USE_RGB
#define CAM_FORMAT V4L2_PIX_FMT_RGB565 
#define GBM_FORMAT GBM_FORMAT_RGB565
#define DRM_FORMAT DRM_FORMAT_RGB565
#else
#define CAM_FORMAT V4L2_PIX_FMT_YVYU
// (v4l2_fourcc('B','G','R','A'))
#define GBM_FORMAT GBM_FORMAT_YUYV
#define DRM_FORMAT DRM_FORMAT_YUYV
#endif


#include <fcntl.h>
#include <errno.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>



struct decoder {
	pthread_t           cam_thread;
    pthread_mutex_t     cam_mutex;

	uint32_t            format;

    const struct gbm   *gbm;
	const struct egl   *egl;
	unsigned            frame;

	EGLImage            last_frame;

    int cam_fd;
    uint8_t* cam_buffers[6];
    int cam_buffer_index;
    int cam_width;
    int cam_height;
    int cam_bytesperline;

};


static void
set_last_frame(struct decoder *dec, EGLImage frame)
{
	if (dec->last_frame)
		dec->egl->eglDestroyImageKHR(dec->egl->display, dec->last_frame);
	dec->last_frame = frame;
}


uint8_t *cam_read_sample(struct decoder *dec) {
#if 0
    struct v4l2_buffer buffinfo;

    fd_set fds;
    struct timeval tv;
    int r;
    while (1) {

        FD_ZERO(&fds);
        FD_SET(dec->cam_fd, &fds);
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(dec->cam_fd + 1, &fds, NULL, NULL, &tv);
        if (r == -1) {
            if (errno = EINTR)
                continue;
            fprintf(stderr, "%s:%i: Call to select() failed\n", __FILE__, __LINE__);
            return 0;
        }
        if (r == 0) {
            fprintf(stderr, "%s:%i: Call to select() timeout\n", __FILE__, __LINE__);
            continue;
        }

        if (!FD_ISSET(dec->cam_fd, &fds))
            continue;

        memset(&buffinfo, 0, sizeof(buffinfo));
        buffinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffinfo.memory = V4L2_MEMORY_MMAP;
        if (ioctl(dec->cam_fd, VIDIOC_DQBUF, &buffinfo) == -1) {
            if (errno == EAGAIN)
                continue;
            fprintf(stderr, "%s:%i: Unable to dequeue buffer\n", __FILE__, __LINE__);
            return 0;
        }

        dec->cam_buffer_index = buffinfo.index;
        return dec->cam_buffers[dec->cam_buffer_index];
    }
#else
    pthread_mutex_lock(&dec->cam_mutex);
    return dec->cam_buffers[dec->cam_buffer_index];
#endif

}

int cam_free_sample(struct decoder *dec) {
#if 0
    struct v4l2_buffer buffinfo;

    memset(&buffinfo, 0, sizeof(buffinfo));
    buffinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffinfo.memory = V4L2_MEMORY_MMAP;
    buffinfo.index = dec->cam_buffer_index;
    if (ioctl(dec->cam_fd, VIDIOC_QBUF, &buffinfo) == -1) {
        fprintf(stderr, "%s:%i: Unable to enqueue buffer\n", __FILE__, __LINE__);
        return -1;
    }
#endif
    pthread_mutex_unlock(&dec->cam_mutex);
    return 0;
}


static void *
cam_thread_func(void *args)
{
#if 1
    struct v4l2_buffer buffinfo;
	struct decoder *dec = args;

    while (1) {
        fd_set fds;
        struct timeval tv;
        int r;

        FD_ZERO(&fds);
        FD_SET(dec->cam_fd, &fds);
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(dec->cam_fd + 1, &fds, NULL, NULL, &tv);
        if (r == -1) {
            if (errno = EINTR)
                continue;
            fprintf(stderr, "%s:%i: Call to select() failed\n", __FILE__, __LINE__);
            return 0;
        }
        if (r == 0) {
            fprintf(stderr, "%s:%i: Call to select() timeout\n", __FILE__, __LINE__);
            continue;
        }

        if (!FD_ISSET(dec->cam_fd, &fds))
            continue;

        memset(&buffinfo, 0, sizeof(buffinfo));
        buffinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffinfo.memory = V4L2_MEMORY_MMAP;
        if (ioctl(dec->cam_fd, VIDIOC_DQBUF, &buffinfo) == -1) {
            if (errno == EAGAIN)
                continue;
            fprintf(stderr, "%s:%i: Unable to dequeue buffer\n", __FILE__, __LINE__);
            return 0;
        }

   	    // TODO make proper locking
        pthread_mutex_lock(&dec->cam_mutex);

        dec->cam_buffer_index = buffinfo.index;
        //show(dec, dec->cam_buffers[dec->cam_buffer_index]);
	    pthread_mutex_unlock(&dec->cam_mutex);

        memset(&buffinfo, 0, sizeof(buffinfo));
        buffinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffinfo.memory = V4L2_MEMORY_MMAP;
        buffinfo.index = dec->cam_buffer_index;
        if (ioctl(dec->cam_fd, VIDIOC_QBUF, &buffinfo) == -1) {
            fprintf(stderr, "%s:%i: Unable to enqueue buffer\n", __FILE__, __LINE__);
            return 0;
        }
    }

#endif
	return NULL;
}

struct decoder* cam_init(const struct egl *egl, const struct gbm *gbm, const char *filename)
{
	struct decoder *dec;

	if (egl_check(egl, eglCreateImageKHR) ||
	    egl_check(egl, eglDestroyImageKHR))
		return NULL;

	dec = calloc(1, sizeof(*dec));
    if (!dec) {
        return 0;
    }
	dec->gbm = gbm;
	dec->egl = egl;

    struct v4l2_requestbuffers reqbuf;
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_buffer buffinfo;
    enum v4l2_buf_type bufftype;
    //int cam_buffer_size;
    int cam_fd;

    /* Setting camera */
    cam_fd = open(filename, O_RDWR | O_NONBLOCK, 0);
    if (!cam_fd) {
        fprintf(stderr, "%s:%i: Couldn't open device\n", __FILE__, __LINE__);
        return 0;
    }
    if (ioctl(cam_fd, VIDIOC_QUERYCAP, &cap)) {
        fprintf(stderr, "%s:%i: Couldn't retreive device capabilities\n", __FILE__, __LINE__);
        return 0;
    }
    if ((cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0) {
        fprintf(stderr, "%s:%i: Device is not a capture device\n", __FILE__, __LINE__);
        return 0;
    }
    if ((cap.capabilities & V4L2_CAP_STREAMING) == 0) {
        fprintf(stderr, "%s:%i: Device is not available for streaming", __FILE__, __LINE__);
        return 0;
    }

    /* Set image format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = CAM_WIDTH;
    fmt.fmt.pix.height = CAM_HEIGHT;
    fmt.fmt.pix.pixelformat = CAM_FORMAT;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(cam_fd, VIDIOC_S_FMT, &fmt) == -1) {
        fprintf(stderr, "%s:%i: Unable to set image format\n", __FILE__, __LINE__);
        return 0;
    }
    //cam_buffer_size = fmt.fmt.pix.sizeimage;
    dec->cam_fd = cam_fd;
    dec->cam_width = fmt.fmt.pix.width;
    dec->cam_height = fmt.fmt.pix.height;
    dec->cam_bytesperline = fmt.fmt.pix.bytesperline;

    /* Request buffers */
    memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.count = CAM_BUFFERS;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(cam_fd, VIDIOC_REQBUFS, &reqbuf) == -1) {
        fprintf(stderr, "%s:%i: Mmap streaming not supported\n", __FILE__, __LINE__);
        return 0;
    }
    if (reqbuf.count < CAM_BUFFERS) {
        fprintf(stderr, "%s:%i: Not all requared buffers are allocated\n", __FILE__, __LINE__);
        return 0;
    }
    printf("Camera: resolution %dx%d with %dbpp buffer count %d\n\r",
        dec->cam_width, dec->cam_height,
        8 * dec->cam_bytesperline / dec->cam_width, reqbuf.count);

    /* Query and Mmap buffers */
    for (unsigned int i = 0; i < reqbuf.count; i++) {
        memset(&buffinfo, 0, sizeof(buffinfo));
        buffinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffinfo.memory = V4L2_MEMORY_MMAP;
        buffinfo.index = i;

        if (ioctl(cam_fd, VIDIOC_QUERYBUF, &buffinfo) == -1) {
            fprintf(stderr, "%s:%i: Unable to query buffers\n", __FILE__, __LINE__);
            return 0;
        }

        dec->cam_buffers[i] = mmap(NULL, buffinfo.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam_fd, buffinfo.m.offset);

        if (dec->cam_buffers[i] == MAP_FAILED) {
            fprintf(stderr, "%s:%i: Unable to enqueue buffers\n", __FILE__, __LINE__);
            return 0;
        }
    }

    /* Enqueue buffers */
    for (unsigned int i = 0; i < reqbuf.count; i++) {
        memset(&buffinfo, 0, sizeof(buffinfo));
        buffinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffinfo.memory = V4L2_MEMORY_MMAP;
        buffinfo.index = i;

        if (ioctl(cam_fd, VIDIOC_QBUF, &buffinfo) == -1) {
            fprintf(stderr, "%s:%i: Unable to enqueue buffers\n", __FILE__, __LINE__);
            return 0;
        }
    }

    /* Start Streaming */
    bufftype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(cam_fd, VIDIOC_STREAMON, &bufftype) == -1) {
        fprintf(stderr, "%s:%i: Unable to start streaming\n", __FILE__, __LINE__);
        return 0;
    }

    pthread_create(&dec->cam_thread, NULL, cam_thread_func, dec);

    return dec;
}

void cam_deinit(struct decoder *dec)
{
	set_last_frame(dec, NULL);
	pthread_join(dec->cam_thread, 0);
	free(dec);
}

//static const uint32_t texw = 512, texh = 512;

static int get_fd_rgba(uint32_t *pstride, uint64_t *modifier, uint8_t *src, unsigned int w, unsigned int h, unsigned int ps)
{
	struct gbm_bo *bo;
	void *map_data = NULL;
	uint32_t stride;
	uint8_t *map;
	int fd;

	/* NOTE: do not actually use GBM_BO_USE_WRITE since that gets us a dumb buffer: */
	//bo = gbm_bo_create(gl.gbm->dev, texw, texh, GBM_FORMAT_RGB565, GBM_BO_USE_LINEAR);
	//map = gbm_bo_map(bo, 0, 0, texw, texh, GBM_BO_TRANSFER_WRITE, &stride, &map_data);
    bo = gbm_bo_create(gl.gbm->dev, w, h, GBM_FORMAT, GBM_BO_USE_LINEAR);
	map = gbm_bo_map(bo, 0, 0, w, h, GBM_BO_TRANSFER_WRITE, &stride, &map_data);

    memcpy(&map[stride], src, h * w * ps);

    gbm_bo_unmap(bo, map_data);

	fd = gbm_bo_get_fd(bo);

	//if (gbm_bo_get_modifier) {
	//	*modifier = gbm_bo_get_modifier(bo);
    //} else {
		*modifier = DRM_FORMAT_MOD_LINEAR;
    //}

	/* we have the fd now, no longer need the bo: */
	gbm_bo_destroy(bo);

	*pstride = stride;

	return fd;
}


static EGLImage
buffer_to_image(struct decoder *dec, uint8_t* src)
{
	EGLImage image;

    uint32_t stride;
	uint64_t modifier;
	int fd = get_fd_rgba(&stride, &modifier, src, dec->cam_width, dec->cam_height, 2);
    
	EGLint attr[] = {
		EGL_WIDTH, dec->cam_width,
		EGL_HEIGHT, dec->cam_height,
		EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT,
		EGL_DMA_BUF_PLANE0_FD_EXT, fd,
		EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
		EGL_DMA_BUF_PLANE0_PITCH_EXT, stride,
		EGL_NONE, EGL_NONE,	/* modifier lo */
		EGL_NONE, EGL_NONE,	/* modifier hi */
		EGL_NONE
	};

	if (egl->modifiers_supported &&
	    modifier != DRM_FORMAT_MOD_INVALID) {
		unsigned size =  ARRAY_SIZE(attr);
		attr[size - 5] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT;
		attr[size - 4] = modifier & 0xFFFFFFFF;
		attr[size - 3] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT;
		attr[size - 2] = modifier >> 32;
	}

	glGenTextures(1, &gl.tex);

	image = egl->eglCreateImageKHR(dec->egl->display, EGL_NO_CONTEXT,
			EGL_LINUX_DMA_BUF_EXT, NULL, attr);

    close(fd);

	return image;
}


static EGLImage
cam_video_frame(struct decoder *dec)
{
	EGLImage   frame = NULL;

    uint8_t *buf = cam_read_sample(dec);
	if (!buf) {
		printf("got no camera sample\n");
		return NULL;
	}

	frame = buffer_to_image(dec, buf);

    cam_free_sample(dec);

	// TODO in the zero-copy dmabuf case it would be nice to associate
	// the eglimg w/ the buffer to avoid recreating it every frame..

	set_last_frame(dec, frame);

	dec->frame++;

	return frame;
}



static void draw_cube_video(unsigned i)
{
	ESMatrix modelview;
	EGLImage frame;

	if (gl.last_fence) {
		egl->eglClientWaitSyncKHR(egl->display, gl.last_fence, 0, EGL_FOREVER_KHR);
		egl->eglDestroySyncKHR(egl->display, gl.last_fence);
		gl.last_fence = NULL;
	}

	frame = cam_video_frame(gl.decoder);

	glUseProgram(gl.blit_program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_EXTERNAL_OES, gl.tex);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	egl->glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, frame);

	/* clear the color buffer */
	glClearColor(0.5, 0.5, 0.5, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(gl.blit_program);
	glUniform1i(gl.blit_texture, 0); /* '0' refers to texture unit 0. */
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glUseProgram(gl.program);

	esMatrixLoadIdentity(&modelview);
	esTranslate(&modelview, 0.0f, 0.0f, -8.0f);
	esRotate(&modelview, 45.0f + (0.25f * i), 1.0f, 0.0f, 0.0f);
	esRotate(&modelview, 45.0f - (0.5f * i), 0.0f, 1.0f, 0.0f);
	esRotate(&modelview, 10.0f + (0.15f * i), 0.0f, 0.0f, 1.0f);

	ESMatrix projection;
	esMatrixLoadIdentity(&projection);
	esFrustum(&projection, -2.1f, +2.1f, -2.1f * gl.aspect, +2.1f * gl.aspect, 6.0f, 10.0f);

	ESMatrix modelviewprojection;
	esMatrixLoadIdentity(&modelviewprojection);
	esMatrixMultiply(&modelviewprojection, &modelview, &projection);

	float normal[9];
	normal[0] = modelview.m[0][0];
	normal[1] = modelview.m[0][1];
	normal[2] = modelview.m[0][2];
	normal[3] = modelview.m[1][0];
	normal[4] = modelview.m[1][1];
	normal[5] = modelview.m[1][2];
	normal[6] = modelview.m[2][0];
	normal[7] = modelview.m[2][1];
	normal[8] = modelview.m[2][2];

	glUniformMatrix4fv(gl.modelviewmatrix, 1, GL_FALSE, &modelview.m[0][0]);
	glUniformMatrix4fv(gl.modelviewprojectionmatrix, 1, GL_FALSE, &modelviewprojection.m[0][0]);
	glUniformMatrix3fv(gl.normalmatrix, 1, GL_FALSE, normal);
	glUniform1i(gl.texture, 0); /* '0' refers to texture unit 0. */

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);
	glDrawArrays(GL_TRIANGLE_STRIP, 8, 4);
	glDrawArrays(GL_TRIANGLE_STRIP, 12, 4);
	glDrawArrays(GL_TRIANGLE_STRIP, 16, 4);
	glDrawArrays(GL_TRIANGLE_STRIP, 20, 4);

	gl.last_fence = egl->eglCreateSyncKHR(egl->display, EGL_SYNC_FENCE_KHR, NULL);
}

const struct egl * init_cube_camera(const struct gbm *gbm, int samples)
{
	int ret;

	ret = init_egl(&gl.egl, gbm, samples);
	if (ret)
		return NULL;

	if (egl_check(&gl.egl, glEGLImageTargetTexture2DOES) ||
	    egl_check(egl, eglCreateSyncKHR) ||
	    egl_check(egl, eglDestroySyncKHR) ||
	    egl_check(egl, eglClientWaitSyncKHR))
		return NULL;

	gl.decoder = cam_init(&gl.egl, gbm, "/dev/video0"); // TODO filenames
	if (!gl.decoder) {
		printf("cannot create video decoder\n");
		return NULL;
	}

	gl.aspect = (GLfloat)(gbm->height) / (GLfloat)(gbm->width);
	gl.gbm = gbm;

	ret = create_program(blit_vs, blit_fs);
	if (ret < 0)
		return NULL;

	gl.blit_program = ret;

	glBindAttribLocation(gl.blit_program, 0, "in_position");
	glBindAttribLocation(gl.blit_program, 1, "in_TexCoord");

	ret = link_program(gl.blit_program);
	if (ret)
		return NULL;

	gl.blit_texture = glGetUniformLocation(gl.blit_program, "uTex");

	ret = create_program(vertex_shader_source, fragment_shader_source);
	if (ret < 0)
		return NULL;

	gl.program = ret;

	glBindAttribLocation(gl.program, 0, "in_position");
	glBindAttribLocation(gl.program, 1, "in_TexCoord");
	glBindAttribLocation(gl.program, 2, "in_normal");

	ret = link_program(gl.program);
	if (ret)
		return NULL;

	gl.modelviewmatrix = glGetUniformLocation(gl.program, "modelviewMatrix");
	gl.modelviewprojectionmatrix = glGetUniformLocation(gl.program, "modelviewprojectionMatrix");
	gl.normalmatrix = glGetUniformLocation(gl.program, "normalMatrix");
	gl.texture   = glGetUniformLocation(gl.program, "uTex");

	glViewport(0, 0, gbm->width, gbm->height);
	glEnable(GL_CULL_FACE);

	gl.positionsoffset = 0;
	gl.texcoordsoffset = sizeof(vVertices);
	gl.normalsoffset = sizeof(vVertices) + sizeof(vTexCoords);

	glGenBuffers(1, &gl.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, gl.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vVertices) + sizeof(vTexCoords) + sizeof(vNormals), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, gl.positionsoffset, sizeof(vVertices), &vVertices[0]);
	glBufferSubData(GL_ARRAY_BUFFER, gl.texcoordsoffset, sizeof(vTexCoords), &vTexCoords[0]);
	glBufferSubData(GL_ARRAY_BUFFER, gl.normalsoffset, sizeof(vNormals), &vNormals[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (const GLvoid *)(intptr_t)gl.positionsoffset);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (const GLvoid *)(intptr_t)gl.texcoordsoffset);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (const GLvoid *)(intptr_t)gl.normalsoffset);
	glEnableVertexAttribArray(2);

	glGenTextures(1, &gl.tex);

	gl.egl.draw = draw_cube_video;

	return &gl.egl;
}
