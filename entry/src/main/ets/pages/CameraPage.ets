import { CameraView } from '../camera/CameraView'
import { IModelType, modelList } from "../model/ModelList"
import { BOTTOM_HEIGHT } from '../utils/ConstantKey'
import { MenuOption } from "../views/MenuOption"
import { NavBar } from "../views/NavBar"
import { image } from '@kit.ImageKit'
import { NNCameraViewController } from '../camera/NNCameraViewController'
import tncnn from 'libtncnn.so'
import { drawBox, IBoxInfo } from '../utils/DrawUtils'
import { resourceManager } from '@kit.LocalizationKit'
import { IConfigType, IOptionType } from '../types/Types'
import { promptAction } from '@kit.ArkUI'
import { OptionDialogContent } from '../views/OptionDialogContent'
import { copyRawfileToSanbox } from '../utils/FileUtils'
import application from '@ohos.app.ability.application'
import LoadingDialog from '@lyb/loading-dialog'
import { taskpool } from '@kit.ArkTS'

@Entry
@ComponentV2
struct CameraPage {
  @Local bottomHeight: number = AppStorage.get(BOTTOM_HEIGHT) || 0
  @Provider('currentModel') currentModel: IModelType = modelList[0]
  @Provider('currentType') currentType: string = 'CPU'
  @Provider('currentCore') currentCore: string = '全部'
  @Provider('currentThread') currentThread: string = '自动'
  option: IOptionType = {
    // net 的参数配置
    mempool: true,
    winograd: true,
    sgemm: true,
    pack4: true,
    bf16Storage: false,
    fp16Packed: true,
    fp16Storage: true,
    fp16Arithmetic: true,
    lightMode: true
  }
  config: IConfigType = {
    isGPU: false,
    core: 0,
    thread: 0
  }
  @Local pixelMap: image.PixelMap | undefined // 优先显示这个图片，有绘制的框
  @Local imageWidth: number = 0 // 图片尺寸
  @Local imageHeight: number = 0 // 图片尺寸
  dialogId: number = 0
  nnCVController: NNCameraViewController = new NNCameraViewController()
  isRunning: boolean = false
  resMgr: resourceManager.ResourceManager | undefined
  modelChange: boolean = false
  paramChange: boolean = false
  delayRun: boolean = false

  @Monitor('currentModel')
  monitorConfigModel(monitor: IMonitor) {
    // 先这么无脑监听吧
    // this.copyAndInit()
    this.modelChange = true
  }

  @Monitor('currentType')
  monitorConfigType(monitor: IMonitor) {
    // 先这么无脑监听吧
    if (monitor.value()?.now) {
      this.config.isGPU = monitor.value()?.now == 'GPU'
    }
    // this.initModel()
    this.paramChange = true
  }

  @Monitor('currentCore')
  monitorConfigCore(monitor: IMonitor) {
    // 先这么无脑监听吧
    if (monitor.value()?.now) {
      this.config.core = (monitor.value()?.now == '全部') ? 0 : (monitor.value()?.now == '大核' ? 2 : 1)
    }
    // this.initModel()
    this.paramChange = true
  }

  @Monitor('currentThread')
  monitorConfigThread(monitor: IMonitor) {
    // 先这么无脑监听吧
    if (monitor.value()?.now) {
      const threadStr: string = monitor.value()?.now as string || '自动'
      this.config.thread = Number(threadStr == '自动' ? 0 : threadStr.split(' ')[threadStr.split(' ').length - 1])
    }
    // this.initModel()
    this.paramChange = true
  }

  handlerClickChangeCamera() {
    promptAction.showToast({ message: '还没写，意外吧' })
  }

  /**
   * 打开配置参数
   */
  handlerClickOption() {
    if (this.dialogId) {
      promptAction.closeCustomDialog(this.dialogId)
    }
    promptAction.openCustomDialog({
      width: 300,
      height: 450,
      cornerRadius: 10,
      autoCancel: false,
      builder: () => this.builderDialogContent()
    })
      .then((dialogId: number) => {
        this.dialogId = dialogId
      })
      .catch((error: Error) => {
        console.log('打开失败了')
      })
  }

  /**
   * 初始化模型
   */
  initModel() {
    let fileDir = getContext().getApplicationContext().filesDir
    // console.log('沙盒路径:' + fileDir)
    if (this.currentModel.name == 'YOLOv4-tiny') {
      const r = tncnn.yolov4_tiny_init(this.resMgr, fileDir + '/models', this.option, this.config)
    } else if (this.currentModel.name == 'nanodet-m') {
      const r = tncnn.nanodet_init(this.resMgr, fileDir + '/models', this.option, this.config)
    }
  }

  /**
   * 复制模型到沙盒并初始化
   */
  async copyAndInit(success: () => void) {
    await LoadingDialog.showLoading('初始化中...')

    console.log(this.currentModel.name)
    // 复制模型到沙盒中(直接加载rawfile目前很麻烦)
    copyRawfileToSanbox(getContext(), this.resMgr!, 'models', this.currentModel.param, () => {
      copyRawfileToSanbox(getContext(), this.resMgr!, 'models', this.currentModel.bin, () => {
        this.initModel()
        LoadingDialog.hide()
        success && success()
      })
    })

  }

  /**
   * 刚打开界面的初始化
   */
  enterInitModel() {
    try {
      application.createModuleContext(getContext(), 'tncnn')
        .then((value: Context) => {
          this.resMgr = value.resourceManager
          this.copyAndInit(() => {
          })
        })
        .catch((e: ESObject) => {
          console.log('获取 resourceManager 异常')
        })
        .finally(() => {

        })
    } catch (e) {
      console.log('加载模型异常:' + e)
    }
  }

  handlerImageArrival(pixelMap: image.PixelMap, width: number, height: number) {
    // console.log(`拿到数据啦 ${width}x${height}`)
    this.imageWidth = width
    this.imageHeight = height

    if (this.isRunning) {
      return
    }
    this.isRunning = true

    try {
      if (this.delayRun) {
        this.isRunning = false
        return
      }
      if (this.modelChange) {
        this.delayRun = true
        this.copyAndInit(() => {
          setTimeout(() => { // 太快会崩...
            this.modelChange = false
            this.delayRun = false
          }, 300)
        })
        this.isRunning = false
        return
      }

      if (this.paramChange) {
        this.initModel()
        this.paramChange = false
      }
    } catch (e) {
    }

    let pixelSize = pixelMap.getPixelBytesNumber()
    console.log("pixel size: " + pixelSize)
    let bufferPixel = new ArrayBuffer(pixelSize)
    pixelMap.readPixelsToBufferSync(bufferPixel)

    // 识别
    let runTest: taskpool.Task = new taskpool.Task(runModelFun, pixelMap, this.currentModel.name,
      bufferPixel, width, height)
    taskpool.execute(runTest, taskpool.Priority.HIGH)
      .then((value: Object) => {
        this.pixelMap = value as image.PixelMap
      })
      .catch((e: ESObject) => {

      })
      .finally(() => {
        this.isRunning = false
      })
  }

  onPageShow(): void {
    this.nnCVController.onPageShow()
  }

  onPageHide(): void {
    this.nnCVController.onPageHide()
  }

  aboutToAppear(): void {
    this.enterInitModel()
  }

  aboutToDisappear(): void {
    this.nnCVController.releaseCamera()
  }

  build() {
    Column() {
      NavBar({ title: 'pnnx 很难用吗' })

      Column() {
        this.builderTopButton()
        this.builderCenterMenu()
      }
      .width('100%')
      .backgroundColor('#CCCCCC')
      .padding({ bottom: 7 })

      Column() {
        this.builderBottomContent()
      }
      .width('100%')
      .height('100%')
      .layoutWeight(1)
    }
    .height('100%')
    .width('100%')
    .backgroundColor($r('app.color.page_background_color'))
  }

  @Builder
  builderTopButton() {
    Row({ space: 7 }) {
      Button('切换镜头')
        .layoutWeight(1)
        .backgroundColor($r('app.color.color_primary'))
        .visibility(Visibility.None)
        .onClick(() => {
          this.handlerClickChangeCamera()
        })
      Button('修改参数')
        .layoutWeight(1)
        .backgroundColor($r('app.color.color_primary'))
        .onClick(() => {
          this.handlerClickOption()
        })
    }
    .width('100%')
    .padding(7)
  }

  @Builder
  builderCenterMenu() {
    Row() {
      MenuOption()
    }
    .width('100%')
  }

  @Builder
  builderBottomContent() {
    Column() {
      Image(this.pixelMap)
        .width('100%')
        .height('100%')
        .layoutWeight(1)
        .objectFit(ImageFit.Contain)

      Row() {
        CameraView({
          nnCVController: this.nnCVController,
          imageArrival: (pixelMap: image.PixelMap, width: number, height: number) => {
            this.handlerImageArrival(pixelMap, width, height)
          }
        })
          .width(120)
          .height(150)
          .borderRadius(7)
          .clip(true)

        Column() {
          Column() {
            Text(`图像尺寸: ${this.imageWidth}x${this.imageHeight}`)
          }
          .width('100%')
          .height('100%')
          .borderRadius(7)
          .justifyContent(FlexAlign.Start)
          .alignItems(HorizontalAlign.Start)
          .backgroundColor('#CCCCCC')
          .padding(5)
        }
        .width('100%')
        .height('100%')
        .layoutWeight(1)
        .padding({ left: 5 })
      }
      .width('100%')
      .height(150 + this.bottomHeight)
      // .backgroundColor('#CCCCCC')
      .padding({ left: 5, right: 5, bottom: this.bottomHeight })
    }
    .width('100%')
    .height('100%')
    .layoutWeight(1)
  }

  @Builder
  builderDialogContent() {
    OptionDialogContent({
      option: this.option,
      action: (option: IOptionType) => {
        this.option = option
        promptAction.closeCustomDialog(this.dialogId)

        this.initModel()
      }
    })
      .width('100%')
      .height('100%')
  }
}

// 线程方式
@Concurrent
function runModelFun(pixelMap: image.PixelMap, modelName: string, imgData: ArrayBuffer, imgWidth: number,
  imgHeight: number): image.PixelMap {
  // 识别
  if (modelName == 'YOLOv4-tiny') {
    const boxInfos: IBoxInfo[] = tncnn.yolov4_tiny_run(imgData, imgWidth, imgHeight)
    if (pixelMap) {
      pixelMap = drawBox(boxInfos, pixelMap, imgWidth, imgHeight)
    }
  } else if (modelName == 'nanodet-m') {
    const boxInfos: IBoxInfo[] = tncnn.nanodet_run(imgData, imgWidth, imgHeight)
    if (pixelMap) {
      pixelMap = drawBox(boxInfos, pixelMap, imgWidth, imgHeight)
    }
  }
  return pixelMap
}
