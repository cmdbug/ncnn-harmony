import { DeviceUtil, DialogUtil, PhotoHelper } from "@pura/harmony-utils"
import { IModelType, modelList } from "../model/ModelList"
import { MenuOption } from "../views/MenuOption"
import { NavBar } from "../views/NavBar"
import { photoAccessHelper } from "@kit.MediaLibraryKit"
import { BOTTOM_HEIGHT } from "../utils/ConstantKey"
import { OptionDialogContent } from "../views/OptionDialogContent"
import { promptAction } from '@kit.ArkUI'
import application from '@ohos.app.ability.application'
import resourceManager from '@ohos.resourceManager';
import tncnn from 'libtncnn.so'
import { copyRawfileToSanbox, getUriInfo } from '../utils/FileUtils'
import fileIo from '@ohos.file.fs'
import { image } from '@kit.ImageKit'
import { drawBox, IBoxInfo } from "../utils/DrawUtils"
import { IConfigType, IOptionType } from '../types/Types'
import LoadingDialog from '@lyb/loading-dialog'
import { IBenchmarkNcnnType } from '../model/BenchmarkNcnnType'
import { taskpool } from '@kit.ArkTS'


@Entry
@ComponentV2
struct PhotoPage {
  @Local bottomHeight: number = AppStorage.get(BOTTOM_HEIGHT) || 0
  @Provider('currentModel') currentModel: IModelType = modelList[0] // 模型列表
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
  @Local imagePath: string = '' // 图片路径
  @Local pixelMap: image.PixelMap | undefined // 优先显示这个图片，有绘制的框
  @Local imageWidth: number = 0 // 图片尺寸
  @Local imageHeight: number = 0 // 图片尺寸
  dialogId: number = 0
  resMgr: resourceManager.ResourceManager | undefined

  @Monitor('currentModel')
  monitorConfigModel(monitor: IMonitor) {
    // 先这么无脑监听吧
    this.copyAndInit(() => {
    })
  }

  @Monitor('currentType')
  monitorConfigType(monitor: IMonitor) {
    // 先这么无脑监听吧
    if (monitor.value()?.now) {
      this.config.isGPU = monitor.value()?.now == 'GPU'
    }
    this.initModel()
  }

  @Monitor('currentCore')
  monitorConfigCore(monitor: IMonitor) {
    // 先这么无脑监听吧
    if (monitor.value()?.now) {
      this.config.core = (monitor.value()?.now == '全部') ? 0 : (monitor.value()?.now == '大核' ? 2 : 1)
    }
    this.initModel()
  }

  @Monitor('currentThread')
  monitorConfigThread(monitor: IMonitor) {
    // 先这么无脑监听吧
    if (monitor.value()?.now) {
      const threadStr: string = monitor.value()?.now as string || '自动'
      this.config.thread = Number(threadStr == '自动' ? 0 : threadStr.split(' ')[threadStr.split(' ').length - 1])
    }
    this.initModel()
  }

  /**
   * 选择图片
   */
  handlerClickPhoto() {
    PhotoHelper.select({
      MIMEType: photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE,
      maxSelectNumber: 1
    })
      .then((value: string[]) => {
        // console.log(value.toString())
        if (value && value.length <= 0) {
          return
        }

        this.imagePath = value[0]
        this.pixelMap = undefined

        // LoadingDialog.showLoading('处理图片...')
        const promise: Promise<void> = new Promise(async (resolve, reject) => {
          // 获取图片数据
          let asset: photoAccessHelper.PhotoAsset | undefined = await getUriInfo(this.imagePath)
          let size: number = asset?.get(photoAccessHelper.PhotoKeys.SIZE) as number
          let width: number = asset?.get(photoAccessHelper.PhotoKeys.WIDTH) as number
          let height: number = asset?.get(photoAccessHelper.PhotoKeys.HEIGHT) as number
          this.imageWidth = width || 0
          this.imageHeight = height || 0

          let file: fileIo.File = fileIo.openSync(this.imagePath, fileIo.OpenMode.READ_ONLY)
          let buffer = new ArrayBuffer(size)
          let readLen = fileIo.readSync(file.fd, buffer) // 这里的 buffer 不能用
          fileIo.closeSync(file)

          const imageSource: image.ImageSource = image.createImageSource(buffer)
          this.pixelMap = await imageSource.createPixelMap({ desiredPixelFormat: image.PixelMapFormat.RGBA_8888 })
          let pixelSize = this.pixelMap.getPixelBytesNumber()
          console.log("pixel size: " + pixelSize)
          imageSource.release()
          let bufferPixel = new ArrayBuffer(pixelSize)
          this.pixelMap.readPixelsToBufferSync(bufferPixel) // 这里的 buffer 才可以

          // 识别
          this.runModel(bufferPixel, width, height)

          resolve()
        })

        return promise
      })
      .then(() => {

      })
      .catch((error: ESObject) => {
        console.log('发生了点问题:' + error)
      })
      .finally(() => {
        // LoadingDialog.hide()
      })
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
   * 基准测试(2种方式执行，使用线程与不使用线程)
   */
  handlerClickBenchmark() {
    try {
      LoadingDialog.showLoading('正在测试...')

      // 使用线程会更慢
      // let fileDir = getContext().getApplicationContext().filesDir
      // let runTest: taskpool.Task = new taskpool.Task(runBenchmark, fileDir + '/models',
      //   this.currentModel.name,
      //   this.currentModel.param,
      //   this.option,
      //   this.config,
      //   30)
      //
      // taskpool.execute(runTest, taskpool.Priority.HIGH)
      //   .then((value: Object) => {
      //     let result: IBenchmarkNcnnType = value as IBenchmarkNcnnType
      // DialogUtil.showDialog({
      //   title: '测试结果',
      //   buttons: [{
      //     value: '心里有点数了吧',
      //     action: () => {},
      //   }],
      //   message: `测试模型: ${this.currentModel.name}`
      //     + `\n循环次数: ${result.loop}`
      //     + `\n最  小  值: ${result.min} ms`
      //     + `\n最  大  值: ${result.max} ms`
      //     + `\n平  均  值: ${result.avg} ms`
      //     + `\n输入尺寸: ${result.width}x${result.height}`
      //     + `\n编译版本: ${tncnn.ncnn_version()}`
      //     + `\n设备型号: ${DeviceUtil.getProductModel()}`
      //     + `\n软件版本: ${DeviceUtil.getSdkApiVersion()}`
      //     + `\n系统版本: ${DeviceUtil.getOsFullName()}`
      //     + `\n`,
      // })
      //   })
      //   .catch(() => {
      //
      //   })
      //   .finally(() => {
      //     LoadingDialog.hide()
      //   })

      // 使用这个会更快，但卡主线程
      setTimeout(() => {
        const promise: Promise<void> = new Promise(async (resolve, reject) => {
          let fileDir = getContext().getApplicationContext().filesDir
          let result: IBenchmarkNcnnType = tncnn.benchmark_ncnn(
            fileDir + '/models',
            this.currentModel.name,
            this.currentModel.param,
            this.option,
            this.config,
            30
          )
          // console.log(JSON.stringify(result))
          DialogUtil.showDialog({
            title: '测试结果',
            buttons: [{
              value: '心里有点数了吧',
              action: () => {
              },
            }],
            message: `测试模型: ${this.currentModel.name}`
              + `\n循环次数: ${result.loop}`
              + `\n最  小  值: ${result.min} ms`
              + `\n最  大  值: ${result.max} ms`
              + `\n平  均  值: ${result.avg} ms`
              + `\n输入尺寸: ${result.width}x${result.height}`
              + `\n编译版本: ${tncnn.ncnn_version()}`
              + `\n设备型号: ${DeviceUtil.getProductModel()}`
              + `\n软件版本: ${DeviceUtil.getSdkApiVersion()}`
              + `\n系统版本: ${DeviceUtil.getOsFullName()}`
              + `\n`,
          })
          resolve()
        })
        promise
          .then(() => {
          })
          .catch(() => {
          })
          .finally(() => {
            LoadingDialog.hide()
          })
      }, 500)
    } catch (e) {
    }

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
   * 运行模型(可以使用taskpool实现线程功能，不卡主线程)
   * @param imgData
   * @param imgWidth
   * @param imgHeight
   */
  runModel(imgData: ArrayBuffer, imgWidth: number, imgHeight: number) {
    const promise: Promise<void> = new Promise(async (resolve: Function, reject: Function) => {
      await LoadingDialog.showLoading('正在识别...')
      if (this.currentModel.name == 'YOLOv4-tiny') {
        const boxInfos: IBoxInfo[] = tncnn.yolov4_tiny_run(imgData, imgWidth, imgHeight)
        if (this.pixelMap) {
          this.pixelMap = drawBox(boxInfos, this.pixelMap, imgWidth, imgHeight)
        }
      } else if (this.currentModel.name == 'nanodet-m') {
        const boxInfos: IBoxInfo[] = tncnn.nanodet_run(imgData, imgWidth, imgHeight)
        if (this.pixelMap) {
          this.pixelMap = drawBox(boxInfos, this.pixelMap, imgWidth, imgHeight)
        }
      }
      resolve()
    })
    promise
      .then(() => {

      })
      .catch(() => {

      })
      .finally(() => {
        LoadingDialog.hide()
      })
  }

  /**
   * 复制模型到沙盒并初始化
   */
  async copyAndInit(success: () => void) {
    await LoadingDialog.showLoading('初始化中...')

    console.log(this.currentModel.name)
    // 复制模型到沙盒中(直接加载rawfile目前很麻烦)
    // 要异步写入...
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
      // const r = tncnn.yolov4_tiny(getContext().createModuleContext('tncnn').resourceManager)
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

  aboutToAppear(): void {
    this.enterInitModel()
  }

  build() {
    Column() {
      NavBar({ title: 'ncnn 纸糊的吗' })

      Column() {
        this.builderTopButton()
        this.builderCenterMenu()
      }
      .width('100%')
      .backgroundColor('#CCCCCC')
      .padding({ bottom: 7 })

      this.builderBottomContent()
    }
    .height('100%')
    .width('100%')
    .backgroundColor($r('app.color.page_background_color'))
  }

  @Builder
  builderTopButton() {
    Row({ space: 7 }) {
      Button('选择图片')
        .layoutWeight(1)
        .backgroundColor($r('app.color.color_primary'))
        .onClick(() => {
          this.handlerClickPhoto()
        })
      Button('修改参数')
        .layoutWeight(1)
        .backgroundColor($r('app.color.color_primary'))
        .onClick(() => {
          this.handlerClickOption()
        })
      Button('基准测试')
        .layoutWeight(1)
        .backgroundColor($r('app.color.color_primary'))
        .onClick(() => {
          this.handlerClickBenchmark()
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
      if (this.pixelMap || this.imagePath) {
        Column() {
          Image(this.pixelMap || this.imagePath)
            .width('100%')
            .height('100%')
            .objectFit(ImageFit.Contain)
            .layoutWeight(1)
          Text(this.imageWidth + 'x' + this.imageHeight)
        }
        .width('100%')
        .height('100%')
      } else {
        Column() {
          Image($r('app.media.ncnn_icon'))
            .width(75)
            .height(75)
            .opacity(0.8)
          Text('请选择图片')
            .fontColor('#777777')
            .margin({ top: 10 })
        }
        .onClick(() => {
          this.handlerClickPhoto()
        })
      }
    }
    .width('100%')
    .height('100%')
    .layoutWeight(1)
    .justifyContent(FlexAlign.Center)
    .padding({ bottom: this.bottomHeight })
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
function runBenchmark(sanboxPath: string, modelName: string, modelParam: string, option: IOptionType,
  config: IConfigType, loop: number): IBenchmarkNcnnType {
  let result: IBenchmarkNcnnType = tncnn.benchmark_ncnn(sanboxPath, modelName, modelParam, option, config, loop)
  return result;
}
