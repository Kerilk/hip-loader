#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include "hip_dispatch.h"

// Instance layers can be added later
struct _multiplex_s {
	struct _hip_dipatch_s dispatch;
};

struct _hip_device_s {
	struct _hip_driver_s *pDriver;
	struct _multiplex_s   multiplex;
	struct _hip_device_s *pNext;
	hipDevice_t           driverHandle;
	hipDevice_t           loaderHandle;
	int                   deviceIndex;
};


static struct _hip_driver_s  *_driverList     = NULL;
static struct _hip_device_s  *_deviceList     = NULL;
static int                    _hipDeviceCount = 0;
static struct _hip_device_s **_deviceArray    = NULL;
static __thread int           _currentDevice  = 0;
static __thread hipError_t    _lastError      = hipSuccess;
static unsigned int           _flags          = 0;
static pthread_once_t         _initialized    = PTHREAD_ONCE_INIT;

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

#define _HIPLD_RETURN(err)       \
do {                             \
	hipError_t _err = (err); \
	_lastError = _err;       \
	return _err;             \
} while(0)

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
		memcpy(&pDevice->multiplex.dispatch, &pDriver->dispatch, sizeof(pDriver->dispatch));
		pDevice->loaderHandle = -_hipDeviceCount - i;
		pDevice->deviceIndex = _hipDeviceCount + i;
		pDevice->pNext = _deviceList;
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
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipDeviceGet(hipDevice_t* device, int ordinal) {
	_initOnce();
	if (ordinal < 0 || ordinal > _hipDeviceCount)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	if (!device)
		_HIPLD_RETURN(hipErrorInvalidValue);
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	*device = _deviceArray[ordinal]->loaderHandle;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipSetDevice(int deviceId) {
	_initOnce();
	if (deviceId < 0 || deviceId > _hipDeviceCount)
		_HIPLD_RETURN(hipErrorInvalidDevice);
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	_currentDevice = deviceId;
	struct _hip_device_s *device = _deviceArray[_currentDevice];
	_HIPLD_RETURN(device->multiplex.dispatch.hipSetDevice(device->deviceIndex));
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipGetDeviceCount(int* count) {
	_initOnce();
	if (!count)
		_HIPLD_RETURN(hipErrorInvalidValue);
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	*count = _hipDeviceCount;
	_HIPLD_RETURN(hipSuccess);
}

hipError_t
hipGetLastError(void) {
	hipError_t _err = _lastError;
	_lastError = hipSuccess;
	return _err;
}

hipError_t
hipPeekAtLastError(void) {
	return _lastError;
}

hipError_t
hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
	_initOnce();
	if (!device || !prop)
		_HIPLD_RETURN(hipErrorInvalidValue);
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	hipError_t err;
	struct _hip_driver_s *driver = _driverList;
	while (driver) {
		err = driver->dispatch.hipChooseDevice(device, prop);
		if (err == hipSuccess) {
			*device = driver->pDevices[*device].deviceIndex;
			_HIPLD_RETURN(hipSuccess);
		}
	}
	_HIPLD_RETURN(hipErrorInvalidValue);
}

hipError_t
hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
	_initOnce();
	if (!device || !pciBusId)
		_HIPLD_RETURN(hipErrorInvalidValue);
	if (!_hipDeviceCount)
		_HIPLD_RETURN(hipErrorNoDevice);
	hipError_t err;
	struct _hip_driver_s *driver = _driverList;
	while (driver) {
		err = driver->dispatch.hipDeviceGetByPCIBusId(device, pciBusId);
		if (err == hipSuccess) {
			*device = driver->pDevices[*device].deviceIndex;
			_HIPLD_RETURN(hipSuccess);
		}
	}
	_HIPLD_RETURN(hipErrorInvalidValue);
}
