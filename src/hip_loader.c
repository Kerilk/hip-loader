#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include "hip_dispatch.h"


struct _hip_context_s;
struct _hip_context_s {
	struct _multiplex_s   *multiplex;
	hipCtx_t               native;
	struct _hip_device_s  *pDevice;
	struct _hip_context_s *pNext;
};

struct _hip_device_s {
	struct _multiplex_s   *multiplex;
	struct _hip_driver_s  *pDriver;
	struct _hip_device_s  *pNext;
	hipDevice_t            driverHandle;
	hipDevice_t            loaderHandle;
	int                    index;
	struct _hip_context_s  primaryCtx;
};

/* For now wrapping objects, so wrapped objects should contain a native member */

#define _HIPLD_UNWRAP(pStruct) ( (pStruct) ? (pStruct)->native : NULL )

struct _thread_context_s {
	struct _hip_device_s  *_currentDevice;
	hipError_t             _lastError;
	struct _hip_context_s *_ctxStack;
};

static __thread  struct _thread_context_s _thread_context =
 {0, hipErrorNotInitialized, NULL};

static struct _hip_driver_s  *_driverList     = NULL;
static struct _hip_device_s  *_deviceList     = NULL;
static int                    _hipDeviceCount = 0;
static struct _hip_device_s **_deviceArray    = NULL;
static unsigned int           _flags          = 0;
static pthread_once_t         _initialized    = PTHREAD_ONCE_INIT;

static inline int _ctxStackEmpty() {
	return _thread_context._ctxStack == NULL;
}

static inline void
_ctxDeviceSet(struct _hip_device_s *dev) {
	_thread_context._currentDevice = dev;
}

static inline void
_ctxDeviceSetID(int dev) {
	_thread_context._currentDevice = _deviceArray[dev];
}

static inline struct _hip_device_s *
_ctxDeviceGet() {
	return _thread_context._currentDevice;
}

static inline int
_ctxDeviceGetID() {
	return _thread_context._currentDevice->loaderHandle;
}

static inline void
_ctxStackPush(struct _hip_context_s *ctx) {
	ctx->pNext = _thread_context._ctxStack;
	_thread_context._ctxStack = ctx;
}

static inline struct _hip_context_s *
_ctxStackPop() {
	if (!_thread_context._ctxStack)
		return NULL;
	struct _hip_context_s *top = _thread_context._ctxStack;
	_thread_context._ctxStack = top->pNext;
	top->pNext = NULL;
	return top;
}

static inline struct _hip_context_s *
_ctxStackTop() {
	return _thread_context._ctxStack;
}

static void *
_loadLibrary(const char *libraryName) {
	return dlopen(libraryName, RTLD_LAZY|RTLD_LOCAL);
}

static char *_get_next(char *paths) {
	char *next;
	next = strchr(paths, ':');
	if (next) {
		*next = '\0';
		next++;
	}
	return next;
}

#define _HIPLD_RETURN(err)                 \
do {                                       \
	hipError_t _err = (err);           \
	_thread_context._lastError = _err; \
	return _err;                       \
} while(0)

#define _HIPLD_CHECK_ERR(err)        \
do {                                 \
	hipError_t _err = (err);     \
	if(_err != hipSuccess)       \
		_HIPLD_RETURN(_err); \
} while(0)

#define _HIPLD_CHECK_DEVICE()                    \
do {                                             \
	if (!_hipDeviceCount)                    \
		_HIPLD_RETURN(hipErrorNoDevice); \
} while(0)

#define _HIPLD_CHECK_DEVICEID(devid)                  \
do {                                                  \
	if (devid < 0 || devid > _hipDeviceCount)     \
		_HIPLD_RETURN(hipErrorInvalidDevice); \
} while(0)

#define _HIPLD_CHECK_PTR(ptr)                        \
do {                                                 \
	if (!ptr)                                    \
		_HIPLD_RETURN(hipErrorInvalidValue); \
} while(0)

#define _HIPLD_CHECK_CTX(ctx)                          \
do {                                                   \
	if (ctx)                                       \
		_HIPLD_RETURN(hipErrorInvalidContext); \
} while(0)

static inline int _indexToHandle(int index) {
  return -index - 1;
}

static inline int _handleToIndex(int handle) {
  return -handle - 1;
}

static hipError_t
_loadDevices(struct _hip_driver_s *pDriver) {
	pDriver->pDevices = (struct _hip_device_s *)calloc(1, sizeof(struct _hip_device_s) * pDriver->deviceCount);
	if (!pDriver->pDevices)
		return hipErrorOutOfMemory;
	for (int i = 0; i < pDriver->deviceCount; i++) {
		struct _hip_device_s *pDevice = pDriver->pDevices + i;
		// Shouldn't fail assert?
		pDriver->hipDeviceGet(&pDevice->driverHandle, i);
		pDevice->pDriver = pDriver;
		pDevice->multiplex = &pDriver->multiplex;
		pDevice->index = _hipDeviceCount + i;
		pDevice->loaderHandle = _indexToHandle(pDevice->index);
		pDevice->pNext = _deviceList;
		pDevice->primaryCtx.multiplex = &pDriver->multiplex;
		pDevice->primaryCtx.pDevice = pDevice;
		_deviceList = pDevice;
	}
	_hipDeviceCount += pDriver->deviceCount;
	return hipSuccess;
}


/**
 * Load a driver given it's library path, checking driver provide the two apis
 * defined in driver-spec.h, and that getPlatformsExt does indeed return a
 * platform.
 */
static void
_loadDriver(const char *path) {
	struct _hip_driver_s driver;
	struct _hip_driver_s *pDriver = NULL;
	void *lib = _loadLibrary(path);
	if (!lib)
		return;
	driver.hipGetDeviceCount = (hipGetDeviceCount_t *)(intptr_t)dlsym(lib, "hipGetDeviceCount");
	driver.hipDeviceGet = (hipDeviceGet_t *)(intptr_t)dlsym(lib, "hipDeviceGet");
	if (!driver.hipGetDeviceCount || !driver.hipDeviceGet)
		goto error;
	driver.pLibrary = lib;
	if (hipSuccess != driver.hipGetDeviceCount(&driver.deviceCount) || driver.deviceCount == 0)
		goto error;
	if (hipSuccess != _fillDriverDispatch(&driver))
		goto error;
	if (hipSuccess != _loadDevices(&driver))
		goto error;
	pDriver = (struct _hip_driver_s *)calloc(1, sizeof(struct _hip_driver_s) + driver.deviceCount * sizeof(void *));
	if (!pDriver)
		goto error;
	memcpy(pDriver, &driver, sizeof(driver));
	pDriver->pDevices = (struct _hip_device_s *)((intptr_t)pDriver + sizeof(struct _hip_driver_s));
	pDriver->pNext = _driverList;
	_driverList = pDriver;
	return;
error:
	dlclose(lib);
}

static void
_initReal() {
	char *drivers = getenv("DRIVERS");
	if (drivers) {
		char *next_file = drivers;
		while (NULL != next_file && *next_file != '\0') {
			char *cur_file = next_file;
			next_file = _get_next(cur_file);
			_loadDriver(cur_file);
		}
		if (_hipDeviceCount) {
			_deviceArray = (struct _hip_device_s **)calloc(_hipDeviceCount, sizeof(struct _hip_device_s));
			if (!_deviceArray)
				return;
			struct _hip_device_s *pDevice = _deviceList;
			int indx = 0;
			while (pDevice && indx < _hipDeviceCount) {
				_deviceArray[indx] = pDevice;
				indx++;
				pDevice = pDevice->pNext;
			}
		}
	}
}

static inline void
_initOnce(void) {
	pthread_once(&_initialized, _initReal);
}

hipError_t
hipInit(unsigned int flags) {
	_flags = flags;
	_initOnce();
	_HIPLD_CHECK_DEVICE();
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipDeviceGet(hipDevice_t* device, int ordinal) {
	_initOnce();
	_HIPLD_CHECK_DEVICE();
	_HIPLD_CHECK_DEVICEID(ordinal);
	_HIPLD_CHECK_PTR(device);
	*device = _deviceArray[ordinal]->loaderHandle;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipSetDevice(int deviceId) {
	_initOnce();
	_HIPLD_CHECK_DEVICE();
	_HIPLD_CHECK_DEVICEID(deviceId);
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	struct _hip_device_s *device = _deviceArray[deviceId];
	_HIPLD_RETURN(device->multiplex->dispatch.hipSetDevice(device->index));
	_ctxDeviceSet(device);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipGetDeviceCount(int* count) {
	_initOnce();
	_HIPLD_CHECK_PTR(count);
	_HIPLD_CHECK_DEVICE();
	*count = _hipDeviceCount;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipGetLastError(void) {
	hipError_t _err = _thread_context._lastError;
	_thread_context._lastError = hipSuccess;
	return _err;
}

hipError_t
hipPeekAtLastError(void) {
	return _thread_context._lastError;
}

hipError_t
hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
	_initOnce();
	_HIPLD_CHECK_PTR(device);
	_HIPLD_CHECK_PTR(prop);
	_HIPLD_CHECK_DEVICE();
	hipError_t err;
	struct _hip_driver_s *driver = _driverList;
	while (driver) {
		err = driver->multiplex.dispatch.hipChooseDevice(device, prop);
		if (err == hipSuccess) {
			*device = driver->pDevices[*device].index;
			_HIPLD_RETURN(hipSuccess);
		}
	}
	_HIPLD_RETURN(hipErrorInvalidValue);
}

hipError_t
hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
	_initOnce();
	_HIPLD_CHECK_PTR(device);
	_HIPLD_CHECK_PTR(pciBusId);
	_HIPLD_CHECK_DEVICE();
	hipError_t err;
	struct _hip_driver_s *driver = _driverList;
	while (driver) {
		err = driver->multiplex.dispatch.hipDeviceGetByPCIBusId(device, pciBusId);
		if (err == hipSuccess) {
			*device = driver->pDevices[*device].index;
			_HIPLD_RETURN(hipSuccess);
		}
	}
	_HIPLD_RETURN(hipErrorInvalidValue);
}

hipError_t
hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
	_initOnce();
	int index = _handleToIndex(device);
	_HIPLD_CHECK_DEVICEID(index);
	struct _hip_device_s *dev = _deviceList + index;
	_HIPLD_CHECK_ERR(dev->multiplex->dispatch.hipCtxCreate(ctx, flags, device));
	struct _hip_context_s * _hip_context = (struct _hip_context_s *)calloc(1, sizeof(struct _hip_context_s));
	if (!_hip_context) {
		dev->multiplex->dispatch.hipCtxDestroy(*ctx);
		_HIPLD_RETURN(hipErrorOutOfMemory);
	}
	_hip_context->multiplex = dev->multiplex;
	_hip_context->pDevice = dev;
	_hip_context->native = *ctx;
	_ctxStackPush(_hip_context);
	_ctxDeviceSet(dev);
	*ctx = (hipCtx_t)_hip_context;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxDestroy(hipCtx_t ctx) {
	_initOnce();
	_HIPLD_CHECK_CTX(ctx);
	struct _hip_context_s * _hip_context = (struct _hip_context_s *)ctx;
	_HIPLD_CHECK_ERR(_hip_context->multiplex->dispatch.hipCtxDestroy(_hip_context->native));
	if (_ctxStackTop() == _hip_context)
		_ctxStackPop();
	free(_hip_context);
	_hip_context = _ctxStackTop();
	if (_hip_context)
		_ctxDeviceSet(_hip_context->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxPopCurrent(hipCtx_t* ctx) {
	_initOnce();
	struct _hip_context_s * _hip_context = _ctxStackPop();
	_HIPLD_CHECK_CTX(_hip_context);
	_HIPLD_CHECK_ERR(_hip_context->multiplex->dispatch.hipCtxPopCurrent(ctx));
	if (ctx)
		*ctx = (hipCtx_t)_hip_context;
	_hip_context = _ctxStackTop();
	if (_hip_context)
		_ctxDeviceSet(_hip_context->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxPushCurrent(hipCtx_t ctx) {
	_initOnce();
	_HIPLD_CHECK_CTX(ctx);
	struct _hip_context_s * _hip_context = (struct _hip_context_s *)ctx;
	_HIPLD_CHECK_ERR(_hip_context->multiplex->dispatch.hipCtxPushCurrent(_hip_context->native));
	_ctxStackPush(_hip_context);
	_ctxDeviceSet(_hip_context->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxSetCurrent(hipCtx_t ctx) {
	_initOnce();
	_HIPLD_CHECK_CTX(ctx);
	struct _hip_context_s * _hip_context = (struct _hip_context_s *)ctx;
	_HIPLD_CHECK_ERR(_hip_context->multiplex->dispatch.hipCtxSetCurrent(_hip_context->native));
	_ctxStackPop();
	_ctxStackPush(_hip_context);
	_ctxDeviceSet(_hip_context->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxGetCurrent(hipCtx_t* ctx) {
	_initOnce();
	_HIPLD_CHECK_PTR(ctx);
	struct _hip_context_s * _hip_context = _ctxStackTop();
	_HIPLD_CHECK_CTX(_hip_context);
	*ctx = (hipCtx_t)_hip_context;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxGetDevice(hipDevice_t* device) {
	_initOnce();
	_HIPLD_CHECK_PTR(device);
	struct _hip_context_s * _hip_context = _ctxStackTop();
	_HIPLD_CHECK_CTX(_hip_context);
	*device = _ctxDeviceGetID();
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
	_initOnce();
	struct _hip_context_s * _hip_context = (struct _hip_context_s *)ctx;
	_HIPLD_CHECK_CTX(_hip_context);
	_HIPLD_CHECK_ERR(_hip_context->multiplex->dispatch.hipCtxGetApiVersion(_hip_context->native, apiVersion));
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
	_initOnce();
	int index = _handleToIndex(dev);
	_HIPLD_CHECK_DEVICEID(index);
	_HIPLD_CHECK_PTR(pctx);
	struct _hip_device_s *_hip_device = _deviceList + index;
	dev = _hip_device->driverHandle;
	_HIPLD_CHECK_ERR(_hip_device->multiplex->dispatch.hipDevicePrimaryCtxRetain(pctx, dev));
	_hip_device->primaryCtx.native = *pctx;
	*pctx = (hipCtx_t)&_hip_device->primaryCtx;
	_HIPLD_RETURN(hipSuccess);
}


#include "hip_dispatch_stubs.h"
