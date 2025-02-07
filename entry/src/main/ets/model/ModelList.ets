export interface IModelType {
  name: string
  param: string
  bin: string
}

export const modelList: IModelType[] = [
  {
    name: 'YOLOv4-tiny',
    param: 'yolov4-tiny.param',
    bin: 'yolov4-tiny.bin'
  },
  {
    name: 'nanodet-m',
    param: 'nanodet-m.param',
    bin: 'nanodet-m.bin'
  }

]
