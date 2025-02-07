import { image } from '@kit.ImageKit';
import { display } from '@kit.ArkUI';
import { stringToColor } from './ColorUtil';

const labels = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush"
]

export interface IBoxInfo {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
  label: number
}

/**
 * 画框
 */
export function drawBox(boxInfos: IBoxInfo[], imagePixelMap: image.PixelMap, width: number, height: number) {
  const offScreenCanvas = new OffscreenCanvas(width, height)
  const offScreenContext = offScreenCanvas.getContext('2d')
  offScreenContext.drawImage(imagePixelMap, 0, 0, width, height)
  const imageScale = width / px2vp(display.getDefaultDisplaySync().width)
  offScreenContext.textAlign = 'left'
  offScreenContext.font = 16 * imageScale + 'vp'
  offScreenContext.lineWidth = 3 * imageScale

  for (let i = 0; i < boxInfos.length; i++) {
    const boxInfo: IBoxInfo = boxInfos[i]
    offScreenContext.strokeStyle = stringToColor(labels[boxInfo.label])
    offScreenContext.strokeRect(boxInfo.x1, boxInfo.y1, boxInfo.x2 - boxInfo.x1, boxInfo.y2 - boxInfo.y1)

    const label = labels[boxInfo.label] + ' ' + boxInfo.score.toFixed(2)
    offScreenContext.fillStyle = '#222222' // 文字阴影
    offScreenContext.fillText(label, boxInfo.x1 + 2, Math.max(boxInfo.y1 - 5 + 2, 20)) // 文字阴影
    offScreenContext.fillStyle = stringToColor(labels[boxInfo.label])
    offScreenContext.fillText(label, boxInfo.x1, Math.max(boxInfo.y1 - 5, 20))
  }
  return offScreenContext.getPixelMap(0, 0, width, height)
}
