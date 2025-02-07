import resourceManager from '@ohos.resourceManager';

/**
 * ncnn 版本号
 */
export const ncnn_version: () => string;

// --------------------------------------------[ yolo start ]--------------------------------------------
export const yolov4_tiny_init: (
  resMgr: resourceManager.ResourceManager,
  sanboxPath: string,
  option: any,
  config: any
) => string;

export const yolov4_tiny_run: (
  imgData: ArrayBuffer,
  imgWidth: number,
  imgHeight: number
) => any[];

// --------------------------------------------[ yolo end ]--------------------------------------------

// --------------------------------------------[ nanodet start ]--------------------------------------------
// ---------------------------------- 早期版本的 nanodet，新版本的模型有变化
export const nanodet_init: (
  resMgr: resourceManager.ResourceManager,
  sanboxPath: string,
  option: any,
  config: any
) => string;

export const nanodet_run: (
  imgData: ArrayBuffer,
  imgWidth: number,
  imgHeight: number
) => any[];

// --------------------------------------------[ nanodet end ]--------------------------------------------

// --------------------------------------------[ benchmark start ]--------------------------------------------
export const benchmark_ncnn: (
  sanboxPath: string,
  model_name: string,
  param_name: string,
  option: any,
  config: any,
  loop: number
) => any;

// --------------------------------------------[ benchmark end ]--------------------------------------------

