#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include "hip_dispatch.h"

#define _HIPLD_DEFINE_HIP_STRUCT(name)  \
struct name {                           \
	struct _multiplex_s *multiplex; \
}

_HIPLD_DEFINE_HIP_STRUCT(ihipCtx_t);
_HIPLD_DEFINE_HIP_STRUCT(ihipStream_t);
_HIPLD_DEFINE_HIP_STRUCT(ihipModule_t);
_HIPLD_DEFINE_HIP_STRUCT(ihipModuleSymbol_t);
_HIPLD_DEFINE_HIP_STRUCT(ihipMemPoolHandle_t);
_HIPLD_DEFINE_HIP_STRUCT(ihipEvent_t);
_HIPLD_DEFINE_HIP_STRUCT(ihipGraph);
_HIPLD_DEFINE_HIP_STRUCT(hipGraphNode);
_HIPLD_DEFINE_HIP_STRUCT(hipGraphExec);
_HIPLD_DEFINE_HIP_STRUCT(hipUserObject);
_HIPLD_DEFINE_HIP_STRUCT(__hip_texture);
_HIPLD_DEFINE_HIP_STRUCT(__hip_surface);
_HIPLD_DEFINE_HIP_STRUCT(ihiprtcLinkState);
_HIPLD_DEFINE_HIP_STRUCT(_hiprtcProgram);
_HIPLD_DEFINE_HIP_STRUCT(_hipGraphicsResource);
_HIPLD_DEFINE_HIP_STRUCT(ihipMemGenericAllocationHandle);

struct _hip_device_s {
	struct _multiplex_s   *multiplex;
	struct _multiplex_s    mplx;
	struct _hip_driver_s  *pDriver;
	struct _hip_device_s  *pNext;
	hipDevice_t            driverHandle;
	hipDevice_t            loaderHandle;
	int                    driverIndex;
	int                    loaderIndex;
	hipCtx_t               primaryCtx;
};

struct _thread_context_s {
	struct _hip_device_s  *_currentDevice;
	hipError_t             _lastError;
	size_t                 _stackSize;
	size_t                 _stackCapacity;
	hipCtx_t              *_ctxStack;
};

static __thread  struct _thread_context_s _thread_context =
	{NULL, hipErrorNotInitialized, 0, 0, NULL};

static struct _hip_driver_s  *_driverList     = NULL;
static int                    _hipDriverCount = 0;
static struct _hip_device_s  *_deviceList     = NULL;
static int                    _hipDeviceCount = 0;
static struct _hip_device_s **_deviceArray    = NULL;
static unsigned int           _flags          = 0;
static pthread_once_t         _initialized    = PTHREAD_ONCE_INIT;

static inline int _ctxStackEmpty() {
	return _thread_context._stackSize == 0;
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

#define _HIPLD_MIN(a, b) ((a)<(b) ? (a) : (b) )

static inline hipError_t
_ctxStackPush(hipCtx_t ctx) {
	if (_thread_context._stackSize == _thread_context._stackCapacity) {
		size_t newCapa = _thread_context._stackCapacity;
		if (newCapa == 0)
			newCapa = 8;
		else
			newCapa *= 2;
		hipCtx_t * newStack = (hipCtx_t *)realloc(
			_thread_context._ctxStack,
			newCapa * sizeof(hipCtx_t));
		if (!newStack)
			return hipErrorOutOfMemory;
		_thread_context._ctxStack = newStack;
		_thread_context._stackCapacity = newCapa;
	}
	_thread_context._ctxStack[_thread_context._stackSize] = ctx;
	_thread_context._stackSize += 1;
	return hipSuccess;
}

static inline hipCtx_t
_ctxStackPop() {
	if (!_thread_context._stackSize)
		return NULL;
	_thread_context._stackSize -= 1;
	return _thread_context._ctxStack[_thread_context._stackSize];
}

static inline hipCtx_t
_ctxStackTop() {
	if (!_thread_context._stackSize)
		return NULL;
	return _thread_context._ctxStack[_thread_context._stackSize - 1];
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

#define _RETURN(expr)  \
do {                   \
	return (expr); \
} while(0)

#define _HIPLDRTC_RETURN(expr)  \
do {                            \
	return (expr);          \
} while(0)

#define _HIPLDRTC_CHECK_ERR(err)        \
do {                                    \
	hiprtcResult _err = (err);      \
	if(_err != HIPRTC_SUCCESS)      \
		_HIPLDRTC_RETURN(_err); \
} while(0)

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


#define _HIPLD_DISPATCH_TABLE(handle) ((handle)->multiplex->dispatch)
#define _HIPLD_DISPATCH_API(handle, api) _HIPLD_DISPATCH_TABLE(handle).api
#define _HIPLD_DISPATCH(handle, api, ...) _HIPLD_DISPATCH_API(handle, api)(__VA_ARGS__)

static inline int _indexToHandle(int index) {
  return index;
}

static inline int _handleToIndex(int handle) {
  return handle;
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
		memcpy(&pDevice->mplx.dispatch, &pDriver->dispatch, sizeof(struct _hip_dipatch_s));
		pDevice->multiplex = &pDevice->mplx;
		pDevice->mplx.pDevice = pDevice;
		pDevice->mplx.pDriver = pDriver;
		pDevice->pDriver = pDriver;
		pDevice->loaderIndex = _hipDeviceCount + i;
		pDevice->driverIndex = i;
		pDevice->loaderHandle = _indexToHandle(pDevice->loaderIndex);
		pDevice->pNext = _deviceList;
		pDevice->primaryCtx = NULL;
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
	driver.hipGetFunc = (hipGetFunc_t *)(intptr_t)dlsym(lib, "hipGetFunc");
	driver.hipGetDeviceCount = (hipGetDeviceCount_t *)
		_hipld_driver_get_function(&driver, "hipGetDeviceCount");
	driver.hipDeviceGet = (hipDeviceGet_t *)
		_hipld_driver_get_function(&driver, "hipDeviceGet");
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
	_hipDriverCount += 1;
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
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(device, hipSetDevice, device->driverIndex));
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
		err = driver->dispatch.hipChooseDevice(device, prop);
		if (err == hipSuccess) {
			*device = driver->pDevices[*device].loaderIndex;
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
		err = driver->dispatch.hipDeviceGetByPCIBusId(device, pciBusId);
		if (err == hipSuccess) {
			*device = driver->pDevices[*device].loaderIndex;
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
	struct _hip_device_s *dev = _deviceArray[index];
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(dev, hipCtxCreate, ctx, flags, device));
	(*ctx)->multiplex = dev->multiplex;
	_ctxStackPush(*ctx);
	_ctxDeviceSet(dev);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxDestroy(hipCtx_t ctx) {
	_initOnce();
	_HIPLD_CHECK_CTX(ctx);
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(ctx, hipCtxDestroy, ctx));
	if (_ctxStackTop() == ctx)
		_ctxStackPop();
	if (_ctxStackTop())
		_ctxDeviceSet(_ctxStackTop()->multiplex->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxPopCurrent(hipCtx_t* ctx) {
	_initOnce();
	hipCtx_t _hip_context = _ctxStackPop();
	_HIPLD_CHECK_CTX(_hip_context);
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_context, hipCtxPopCurrent, ctx));
	if (ctx)
		*ctx = _hip_context;
	_hip_context = _ctxStackTop();
	if (_hip_context)
		_ctxDeviceSet(_hip_context->multiplex->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxPushCurrent(hipCtx_t ctx) {
	_initOnce();
	_HIPLD_CHECK_CTX(ctx);
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(ctx, hipCtxPushCurrent, ctx));
	_ctxStackPush(ctx);
	_ctxDeviceSet(ctx->multiplex->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxSetCurrent(hipCtx_t ctx) {
	_initOnce();
	_HIPLD_CHECK_CTX(ctx);
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(ctx, hipCtxSetCurrent, ctx));
	_ctxStackPop();
	_ctxStackPush(ctx);
	_ctxDeviceSet(ctx->multiplex->pDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxGetCurrent(hipCtx_t* ctx) {
	_initOnce();
	_HIPLD_CHECK_PTR(ctx);
	*ctx = _ctxStackTop();
	_HIPLD_CHECK_CTX(*ctx);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipCtxGetDevice(hipDevice_t* device) {
	_initOnce();
	_HIPLD_CHECK_PTR(device);
	hipCtx_t _hip_context = _ctxStackTop();
	_HIPLD_CHECK_CTX(_hip_context);
	*device = _ctxDeviceGetID();
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
	_initOnce();
	int index = _handleToIndex(dev);
	_HIPLD_CHECK_DEVICEID(index);
	_HIPLD_CHECK_PTR(pctx);
	struct _hip_device_s *_hip_device = _deviceArray[index];
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_device, hipDevicePrimaryCtxRetain,
		pctx, _hip_device->driverHandle));
	_hip_device->primaryCtx = *pctx;
	_hip_device->primaryCtx->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
	_initOnce();
	_HIPLD_CHECK_PTR(canAccessPeer);
	_HIPLD_CHECK_DEVICEID(deviceId);
	_HIPLD_CHECK_DEVICEID(peerDeviceId);
	if (deviceId == peerDeviceId) {
		*canAccessPeer = 0;
	} else {
		struct _hip_device_s *_hip_device = _deviceArray[deviceId];
		struct _hip_device_s *_hip_peer_device = _deviceArray[peerDeviceId];
		if (_hip_device->pDriver != _hip_peer_device->pDriver)
			*canAccessPeer = 0;
		else {
			_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_device, hipDeviceCanAccessPeer,
				canAccessPeer, _hip_device->driverIndex, _hip_peer_device->driverIndex));
		}
	}
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipDeviceDisablePeerAccess(int peerDeviceId) {
	_initOnce();
	_HIPLD_CHECK_DEVICEID(peerDeviceId);
	struct _hip_device_s *_hip_device = _ctxDeviceGet();
	struct _hip_device_s *_hip_peer_device = _deviceArray[peerDeviceId];
	if (_hip_device->pDriver != _hip_peer_device->pDriver)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_device, hipDeviceDisablePeerAccess,
		_hip_peer_device->driverIndex));
}

hipError_t
hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
	_initOnce();
	_HIPLD_CHECK_DEVICEID(peerDeviceId);
	struct _hip_device_s *_hip_device = _ctxDeviceGet();
	struct _hip_device_s *_hip_peer_device = _deviceArray[peerDeviceId];
	if (_hip_device->pDriver != _hip_peer_device->pDriver)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_device, hipDeviceEnablePeerAccess,
		_hip_peer_device->driverIndex, flags));
}

hipError_t
hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice) {
	_initOnce();
	_HIPLD_CHECK_DEVICEID(srcDevice);
	_HIPLD_CHECK_DEVICEID(dstDevice);
	struct _hip_device_s *_hip_src_device = _deviceArray[srcDevice];
	struct _hip_device_s *_hip_dst_device = _deviceArray[dstDevice];
	if (_hip_src_device->pDriver != _hip_dst_device->pDriver)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_src_device, hipDeviceGetP2PAttribute,
		value, attr, _hip_src_device->driverIndex, _hip_dst_device->driverIndex));
}

hipError_t
hipEventCreate(hipEvent_t* event) {
	_initOnce();
	struct _hip_device_s *_hip_device = _ctxDeviceGet();
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_device, hipEventCreate, event));
	(*event)->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
	_initOnce();
	struct _hip_device_s *_hip_device = _ctxDeviceGet();
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_device, hipEventCreateWithFlags, event, flags));
	(*event)->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount) {
	_initOnce();
	_HIPLD_CHECK_DEVICEID(device1);
	_HIPLD_CHECK_DEVICEID(device2);
	struct _hip_device_s *_hip_device1 = _deviceArray[device1];
	struct _hip_device_s *_hip_device2 = _deviceArray[device2];
	if (_hip_device1->pDriver != _hip_device2->pDriver)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_device1, hipExtGetLinkTypeAndHopCount,
		_hip_device1->driverIndex, _hip_device2->driverIndex, linktype, hopcount));
}

hipError_t
hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int  numDevices, unsigned int  flags)
{
	_initOnce();
	if (!numDevices)
		_HIPLD_RETURN(hipSuccess);
	if (launchParamsList[0].stream)
		_HIPLD_RETURN(_HIPLD_DISPATCH(launchParamsList[0].stream, hipExtLaunchMultiKernelMultiDevice,
			launchParamsList, numDevices, flags));
	else
		_HIPLD_RETURN(_HIPLD_DISPATCH(_ctxDeviceGet(), hipExtLaunchMultiKernelMultiDevice,
			launchParamsList, numDevices, flags));
}

hipError_t hipGetDevice(int* deviceId)
{
	_initOnce();
	if (!deviceId)
		_HIPLD_RETURN(hipErrorInvalidValue);
	*deviceId = _ctxDeviceGet()->loaderIndex;
	_HIPLD_RETURN(hipSuccess);
}

const char*
hipGetErrorName(hipError_t hip_error) {
	_initOnce();
	return _HIPLD_DISPATCH(_ctxDeviceGet(), hipGetErrorName, hip_error);
}

const char*
hipGetErrorString(hipError_t hipError) {
	_initOnce();
	return _HIPLD_DISPATCH(_ctxDeviceGet(), hipGetErrorString, hipError);
}

hipError_t
hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes) {
	_initOnce();
	_HIPLD_CHECK_DEVICEID(dstDeviceId);
	_HIPLD_CHECK_DEVICEID(srcDeviceId);
	struct _hip_device_s *_hip_dst_device = _deviceArray[dstDeviceId];
	struct _hip_device_s *_hip_src_device = _deviceArray[srcDeviceId];
	if (_hip_src_device->pDriver != _hip_dst_device->pDriver)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_dst_device, hipMemcpyPeer,
		dst, _hip_dst_device->driverIndex, src, _hip_src_device->driverIndex, sizeBytes));
}

hipError_t
hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice, size_t sizeBytes, hipStream_t stream __dparm(0)) {
	_initOnce();
	_HIPLD_CHECK_DEVICEID(dstDeviceId);
	_HIPLD_CHECK_DEVICEID(srcDevice);
	struct _hip_device_s *_hip_dst_device = _deviceArray[dstDeviceId];
	struct _hip_device_s *_hip_src_device = _deviceArray[srcDevice];
	if (_hip_src_device->pDriver != _hip_dst_device->pDriver)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	_HIPLD_RETURN(_HIPLD_DISPATCH(_hip_dst_device, hipMemcpyPeerAsync,
		dst, _hip_dst_device->driverIndex, src, _hip_src_device->driverIndex, sizeBytes,
		stream));
}

hipError_t
hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int  numDevices, unsigned int  flags)
{
	_initOnce();
	if (!numDevices)
		_HIPLD_RETURN(hipSuccess);
	_HIPLD_CHECK_PTR(launchParamsList);
	if (launchParamsList[0].stream)
		_HIPLD_RETURN(_HIPLD_DISPATCH(launchParamsList[0].stream, hipLaunchCooperativeKernelMultiDevice,
			launchParamsList, numDevices, flags));
	else
		_HIPLD_RETURN(_HIPLD_DISPATCH(_ctxDeviceGet(), hipLaunchCooperativeKernelMultiDevice,
			launchParamsList, numDevices, flags));
}

hipError_t
hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
	_initOnce();
	unsigned int _count;
	if (!pHipDeviceCount && pHipDevices)
		pHipDeviceCount = &_count;
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_ctxDeviceGet(), hipGLGetDevices,
		pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList));
	if (pHipDevices)
		for (unsigned int i = 0; i < *pHipDeviceCount; i++) {
			pHipDevices[i] = _ctxDeviceGet()->pDriver->pDevices[pHipDevices[i]].loaderIndex;
		}
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
	_initOnce();
	struct _hip_device_s *_hip_device;
	if (stream)
		_hip_device = stream->multiplex->pDevice;
	else
		_hip_device = _ctxDeviceGet ();
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_device, hipStreamGetCaptureInfo_v2,
		stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out));
	if (graph_out && *graph_out)
		(*graph_out)->multiplex = _hip_device->multiplex;
	if (dependencies_out)
		for (size_t i = 0; i < *numDependencies_out; i++)
			(*dependencies_out)[i]->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipMemPoolCreate(hipMemPool_t * mem_pool, const hipMemPoolProps * pool_props)
{
	_initOnce ();
	_HIPLD_CHECK_PTR(pool_props);
	struct _hip_device_s *_hip_device = _ctxDeviceGet ();
	if (pool_props->location.type == hipMemLocationTypeDevice) {
		_HIPLD_CHECK_DEVICEID(pool_props->location.id);
		if (_deviceArray[pool_props->location.id]->pDriver != _hip_device->pDriver)
			_HIPLD_RETURN(hipErrorInvalidDevice);
		((hipMemAccessDesc *)pool_props)->location.id = _deviceArray[pool_props->location.id]->driverIndex;
	}
	hipError_t _hip_err = _HIPLD_DISPATCH(_hip_device, hipMemPoolCreate,
		mem_pool, pool_props);
	if (pool_props->location.type == hipMemLocationTypeDevice)
		((hipMemAccessDesc *)pool_props)->location.id = _hip_device->pDriver->pDevices[pool_props->location.id].loaderIndex;
	_HIPLD_CHECK_ERR(_hip_err);
	if (mem_pool && *mem_pool)
		(*mem_pool)->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc * desc_list, size_t count)
{
	_initOnce ();
	if (!count)
		_HIPLD_RETURN(hipSuccess);
	_HIPLD_CHECK_PTR(desc_list);
	struct _hip_driver_s *_hip_driver = mem_pool->multiplex->pDriver;
	for (size_t i = 0; i < count; i++) {
		if (desc_list[i].location.type == hipMemLocationTypeDevice) {
			_HIPLD_CHECK_DEVICEID(desc_list[i].location.id);
			if (_hip_driver != _deviceArray[desc_list[i].location.id]->pDriver)
				_HIPLD_RETURN(hipErrorInvalidDevice);
		}
	}
	for (size_t i = 0; i < count; i++)
		if (desc_list[i].location.type == hipMemLocationTypeDevice)
			((hipMemAccessDesc *)desc_list)[i].location.id = _deviceArray[desc_list[i].location.id]->driverIndex;
	hipError_t _hip_err = _HIPLD_DISPATCH(mem_pool, hipMemPoolSetAccess,
		mem_pool, desc_list, count);
	for (size_t i = 0; i < count; i++)
		if (desc_list[i].location.type == hipMemLocationTypeDevice)
			((hipMemAccessDesc *)desc_list)[i].location.id = _hip_driver->pDevices[desc_list[i].location.id].loaderIndex;
	_HIPLD_RETURN(_hip_err);
}

hipError_t
hipMemPoolGetAccess(hipMemAccessFlags * flags, hipMemPool_t mem_pool, hipMemLocation * location) {
	_initOnce ();
	_HIPLD_CHECK_PTR(location);
	struct _hip_driver_s *_hip_driver = mem_pool->multiplex->pDriver;
	if (location->type == hipMemLocationTypeDevice) {
		_HIPLD_CHECK_DEVICEID(location->id);
		if (_deviceArray[location->id]->pDriver != _hip_driver)
			_HIPLD_RETURN(hipErrorInvalidDevice);
		location->id = _deviceArray[location->id]->driverIndex;
	}
	hipError_t _hip_err = _HIPLD_DISPATCH(mem_pool, hipMemPoolGetAccess,
		flags, mem_pool, location);
	if (location->type == hipMemLocationTypeDevice)
		location->id = _hip_driver->pDevices[location->id].loaderIndex;
	_HIPLD_RETURN(_hip_err);
}

hipError_t
hipMemCreate(hipMemGenericAllocationHandle_t * handle, size_t size, const hipMemAllocationProp * prop, unsigned long long int flags) {
	_initOnce ();
	_HIPLD_CHECK_PTR(prop);
	struct _hip_device_s *_hip_device = _ctxDeviceGet ();
	struct _hip_driver_s *_hip_driver = _hip_device->pDriver;
	if (prop->location.type == hipMemLocationTypeDevice) {
		_HIPLD_CHECK_DEVICEID(prop->location.id);
		if (_deviceArray[prop->location.id]->pDriver != _hip_driver)
			_HIPLD_RETURN(hipErrorInvalidDevice);
		((hipMemAllocationProp *)prop)->location.id = _deviceArray[prop->location.id]->driverIndex;
	}
	hipError_t _hip_err = _HIPLD_DISPATCH (_hip_device, hipMemCreate,
		handle, size, prop, flags);
	if (prop->location.type == hipMemLocationTypeDevice)
		((hipMemAllocationProp *)prop)->location.id = _hip_driver->pDevices[prop->location.id].loaderIndex;
	_HIPLD_CHECK_ERR(_hip_err);
	if (handle && *handle)
		(*handle)->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
	_initOnce ();
	_HIPLD_CHECK_PTR(location);
	struct _hip_driver_s *_hip_driver = _ctxDeviceGet()->pDriver;
	if (location->type == hipMemLocationTypeDevice) {
		_HIPLD_CHECK_DEVICEID(location->id);
		if (_deviceArray[location->id]->pDriver != _hip_driver)
			_HIPLD_RETURN(hipErrorInvalidDevice);
		((hipMemLocation *)location)->id = _deviceArray[location->id]->driverIndex;
	}
	hipError_t _hip_err = _HIPLD_DISPATCH(_ctxDeviceGet(), hipMemGetAccess,
		flags, location, ptr);
	if (location->type == hipMemLocationTypeDevice)
		((hipMemLocation *)location)->id = _hip_driver->pDevices[location->id].loaderIndex;
	_HIPLD_RETURN(_hip_err);
}

hipError_t
hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option) {
	_initOnce ();
	_HIPLD_CHECK_PTR(prop);
	struct _hip_device_s *_hip_device = _ctxDeviceGet ();
	struct _hip_driver_s *_hip_driver = _hip_device->pDriver;
	if (prop->location.type == hipMemLocationTypeDevice) {
		_HIPLD_CHECK_DEVICEID(prop->location.id);
		if (_deviceArray[prop->location.id]->pDriver != _hip_driver)
			_HIPLD_RETURN(hipErrorInvalidDevice);
		((hipMemAllocationProp *)prop)->location.id = _deviceArray[prop->location.id]->driverIndex;
	}
	hipError_t _hip_err = _HIPLD_DISPATCH (_hip_device, hipMemGetAllocationGranularity,
		granularity, prop, option);
	if (prop->location.type == hipMemLocationTypeDevice)
		((hipMemAllocationProp *)prop)->location.id = _hip_driver->pDevices[prop->location.id].loaderIndex;
	_HIPLD_RETURN(_hip_err);
}

hipError_t
hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
	_initOnce ();
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(handle, hipMemGetAllocationPropertiesFromHandle,
		prop, handle));
	if (prop->location.type == hipMemLocationTypeDevice)
		((hipMemAllocationProp *)prop)->location.id = handle->multiplex->pDriver->pDevices[prop->location.id].loaderIndex;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
	_initOnce ();
	_HIPLD_CHECK_PTR(desc);
	struct _hip_driver_s *_hip_driver = _ctxDeviceGet()->pDriver;
	for (size_t i = 0; i < count; i++) {
		if (desc[i].location.type == hipMemLocationTypeDevice) {
			_HIPLD_CHECK_DEVICEID(desc[i].location.id);
			if (_hip_driver != _deviceArray[desc[i].location.id]->pDriver)
				_HIPLD_RETURN(hipErrorInvalidDevice);
		}
	}
	for (size_t i = 0; i < count; i++)
		if (desc[i].location.type == hipMemLocationTypeDevice)
			((hipMemAccessDesc *)desc)[i].location.id = _deviceArray[desc[i].location.id]->driverIndex;
	hipError_t _hip_err = _HIPLD_DISPATCH(_ctxDeviceGet(), hipMemSetAccess,
		ptr, size, desc, count);
	for (size_t i = 0; i < count; i++)
		if (desc[i].location.type == hipMemLocationTypeDevice)
			((hipMemAccessDesc *)desc)[i].location.id = _hip_driver->pDevices[desc[i].location.id].loaderIndex;
	_HIPLD_RETURN(_hip_err);
}

hipError_t
hipStreamGetCaptureInfo_v2_spt(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
	_initOnce();
	struct _hip_device_s *_hip_device;
	if (stream)
		_hip_device = stream->multiplex->pDevice;
	else
		_hip_device = _ctxDeviceGet ();
	_HIPLD_CHECK_ERR(_HIPLD_DISPATCH(_hip_device, hipStreamGetCaptureInfo_v2,
		stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out));
	if (graph_out && *graph_out)
		(*graph_out)->multiplex = _hip_device->multiplex;
	if (dependencies_out)
		for (size_t i = 0; i < *numDependencies_out; i++)
			(*dependencies_out)[i]->multiplex = _hip_device->multiplex;
	_HIPLD_RETURN(hipSuccess);
}

void **
__hipRegisterFatBinary(const void *Data) {
	_initOnce();
	void *** res = (void***)calloc(_hipDriverCount, sizeof(void**));
	if (!res)
		return NULL;
	struct _hip_driver_s *driver = _driverList;
	for (int i = 0; i < _hipDriverCount; i++) {
		res[i] = driver->dispatch.__hipRegisterFatBinary(Data);
		driver = driver->pNext;
	}
	return (void **)res;
}

void
__hipUnregisterFatBinary(void **modules) {
	_initOnce();
	if (!modules)
		return;
	void *** res = (void ***)modules;
	struct _hip_driver_s *driver = _driverList;
	for (int i = 0; i < _hipDriverCount; i++)
		if (res[i])
			driver->dispatch.__hipUnregisterFatBinary(res[i]);
	free(res);
}

const char*
hipApiName(uint32_t id) {
	_initOnce();
	struct _hip_driver_s *driver = _driverList;
	for (int i = 0; i < _hipDriverCount; i++) {
		const char *name = driver->dispatch.hipApiName(id);
		if (name)
			return name;
	}
	return NULL;
}

hiprtcResult
hiprtcDestroyProgram(hiprtcProgram* prog) {
	_initOnce();
	if (!prog)
		return HIPRTC_ERROR_INVALID_INPUT;
	_HIPLDRTC_RETURN(_HIPLD_DISPATCH(*prog, hiprtcDestroyProgram, prog));
}

/* from hipamd */

static inline uint32_t f32_as_u32(float f) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = f;
  return v.u;
}

static inline float u32_as_f32(uint32_t u) {
  union {
    float f;
    uint32_t u;
  } v;
  v.u = u;
  return v.f;
}

static inline int min(int a, int b) {
  return a < b ? a : b;
}

static inline int max(int a, int b) {
  return a > b ? a : b;
}

static inline int clamp_int(int i, int l, int h) { return min(max(i, l), h); }


// half float, the f16 is in the low 16 bits of the input argument

static inline float __convert_half_to_float(uint32_t a) {
  uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;

  uint32_t v =
      f32_as_u32(u32_as_f32(u) * u32_as_f32(0x77800000U) /*0x1.0p+112f*/) + 0x38000000U;

  u = (a & 0x7fff) != 0 ? v : u;

  return u32_as_f32(u) * u32_as_f32(0x07800000U) /*0x1.0p-112f*/;
}

// float half with nearest even rounding
// The lower 16 bits of the result is the bit pattern for the f16
static inline uint32_t __convert_float_to_half(float a) {
  uint32_t u = f32_as_u32(a);
  int e = (int)((u >> 23) & 0xff) - 127 + 15;
  uint32_t m = ((u >> 11) & 0xffe) | ((u & 0xfff) != 0);
  uint32_t i = 0x7c00 | (m != 0 ? 0x0200 : 0);
  uint32_t n = ((uint32_t)e << 12) | m;
  uint32_t s = (u >> 16) & 0x8000;
  int b = clamp_int(1 - e, 0, 13);
  uint32_t d = (0x1000 | m) >> b;
  d |= (d << b) != (0x1000 | m);
  uint32_t v = e < 1 ? d : n;
  v = (v >> 2) + (((v & 0x7) == 3) | ((v & 0x7) > 5));
  v = e > 30 ? 0x7c00 : v;
  v = e == 143 ? i : v;
  return s | v;
}

__attribute__((weak))
float
__gnu_h2f_ieee(unsigned short h) {
  return __convert_half_to_float((uint32_t)h);
}

__attribute__((weak))
unsigned short
__gnu_f2h_ieee(float f) {
  return (unsigned short)__convert_float_to_half(f);
}

#include "hip_dispatch_stubs.h"
