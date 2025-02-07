import { camera } from '@kit.CameraKit';
import { image } from '@kit.ImageKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { display } from '@kit.ArkUI';
import { NNCameraViewController } from './NNCameraViewController';


@ComponentV2
export struct CameraView {
  private imageReceiver: image.ImageReceiver | undefined = undefined;
  private imageReceiverSurfaceId: string = '';
  private xComponentCtl: XComponentController = new XComponentController();
  private xComponentSurfaceId: string = '';
  @Local imageWidth: number = 960; // 最接近的尺寸
  @Local imageHeight: number = 720; // 最接近的尺寸
  private cameraManager: camera.CameraManager | undefined = undefined;
  private cameras: Array<camera.CameraDevice> | Array<camera.CameraDevice> = [];
  private cameraInput: camera.CameraInput | undefined = undefined;
  private previewOutput1: camera.PreviewOutput | undefined = undefined;
  private previewOutput2: camera.PreviewOutput | undefined = undefined;
  private session: camera.VideoSession | undefined = undefined;
  // 相机控制
  @Param @Require nnCVController: NNCameraViewController = new NNCameraViewController()
  // 预览图片处理
  @Param imageArrival: (pixelMap: image.PixelMap, width: number, height: number) => void =
    (pixelMap: image.PixelMap, width: number, height: number) => {

    }
  cameraID: number = 0

  onPageShow(): void {
    console.info('onPageShow');
    this.initImageReceiver();
    if (this.xComponentSurfaceId !== '') {
      this.initCamera();
    }
  }

  onPageHide(): void {
    console.info('onPageHide');
    this.releaseCamera();
  }

  /**
   * 在非 Entry 中使用
   */
  onPageShowNNCV() {
    console.info('onPageShowNNCV');
    this.initImageReceiver();
    if (this.xComponentSurfaceId !== '') {
      this.initCamera();
    }
  }

  /**
   * 在非 Entry 中使用
   */
  onPageHideNNCV() {
    console.info('onPageHideNNCV');
    this.initImageReceiver();
    if (this.xComponentSurfaceId !== '') {
      this.initCamera();
    }
  }

  /**
   * 获取ImageReceiver的SurfaceId
   */
  async initImageReceiver(): Promise<void> {
    console.log('initImageReceiver')
    if (!this.imageReceiver) {
      // 创建ImageReceiver
      let size: image.Size = { width: this.imageWidth, height: this.imageHeight };
      this.imageReceiver = image.createImageReceiver(size, image.ImageFormat.JPEG, image.PixelMapFormat.NV21);
      // 获取取第一路流SurfaceId
      this.imageReceiverSurfaceId = await this.imageReceiver.getReceivingSurfaceId();
      console.info(`initImageReceiver imageReceiverSurfaceId:${this.imageReceiverSurfaceId}`);
      // 注册监听处理预览流每帧图像数据
      this.onImageArrival(this.imageReceiver);
    }
  }

  /**
   * 注册ImageReceiver图像监听
   */
  onImageArrival(receiver: image.ImageReceiver): void {
    console.log('onImageArrival')
    // 注册imageArrival监听
    receiver.on('imageArrival', () => {
      // console.info('image arrival');
      // 获取图像
      receiver.readNextImage((err: BusinessError, nextImage: image.Image) => {
        if (err || nextImage === undefined) {
          console.error('readNextImage failed');
          return;
        }
        // 解析图像内容
        nextImage.getComponent(image.ComponentType.JPEG, async (err: BusinessError, imgComponent: image.Component) => {
          if (err || imgComponent === undefined) {
            console.error('getComponent failed');
          }
          if (imgComponent.byteBuffer) {
            // 请参考步骤7解析buffer数据，本示例以方式一为例
            let width = nextImage.size.width; // 获取图片的宽
            let height = nextImage.size.height; // 获取图片的高
            let stride = imgComponent.rowStride; // 获取图片的stride
            // console.debug(`getComponent with width:${width} height:${height} stride:${stride}`);
            // stride与width一致
            if (stride == width) {
              let pixelMap = await image.createPixelMap(imgComponent.byteBuffer, {
                size: { height: height, width: width },
                srcPixelFormat: image.PixelMapFormat.NV21,
                pixelFormat: image.PixelMapFormat.RGBA_8888
              })

              pixelMap.rotateSync(90) // 旋转后宽高要换一下
              this.imageArrival && this.imageArrival(pixelMap, height, width) // 回调
            } else {
              // stride与width不一致
              const dstBufferSize = width * height * 1.5 // 以NV21为例（YUV_420_SP格式的图片）YUV_420_SP内存计算公式：长x宽+(长x宽)/2
              const dstArr = new Uint8Array(dstBufferSize)
              for (let j = 0; j < height * 1.5; j++) {
                const srcBuf = new Uint8Array(imgComponent.byteBuffer, j * stride, width)
                dstArr.set(srcBuf, j * width)
              }
              let pixelMap = await image.createPixelMap(dstArr.buffer, {
                size: { height: height, width: width },
                srcPixelFormat: image.PixelMapFormat.NV21,
                pixelFormat: image.PixelMapFormat.RGBA_8888
              })

              pixelMap.rotateSync(90) // 旋转后宽高要换一下
              this.imageArrival && this.imageArrival(pixelMap, height, width) // 回调
            }
          } else {
            console.error('byteBuffer is null');
          }
          // 确保当前buffer没有在使用的情况下，可进行资源释放
          // 如果对buffer进行异步操作，需要在异步操作结束后再释放该资源（nextImage.release()）
          nextImage.release();
          // console.info('image process done');
        })
      })
    })
  }

  private initNNController() {
    this.nnCVController.onPageShow = () => {
      this.onPageShowNNCV()
    }

    this.nnCVController.onPageHide = () => {
      this.onPageHideNNCV()
    }

    this.nnCVController.initCamera = () => {
      this.initCamera()
    }

    this.nnCVController.releaseCamera = () => {
      this.releaseCamera()
    }
  }

  aboutToAppear(): void {
    this.initNNController()
  }

  build() {
    Column() {
      XComponent({
        id: 'componentId',
        type: XComponentType.SURFACE,
        controller: this.xComponentCtl
      })
        .onLoad(async () => {
          console.info('onLoad is called');
          this.xComponentSurfaceId = this.xComponentCtl.getXComponentSurfaceId(); // 获取组件surfaceId
          // 初始化相机，组件实时渲染每帧预览流数据
          this.initCamera()
        })
        .width('100%')
        .height('100%')
    }
    .justifyContent(FlexAlign.Center)
    .height('100%')
    .width('100%')
  }

  // 初始化相机
  async initCamera(): Promise<void> {
    console.info(`initCamera imageReceiverSurfaceId:${this.imageReceiverSurfaceId}`);
    console.info(`initCamera xComponentSurfaceId:${this.xComponentSurfaceId}`);
    try {
      // 获取相机管理器实例
      this.cameraManager = camera.getCameraManager(getContext(this));
      if (!this.cameraManager) {
        console.error('initCamera getCameraManager');
      }
      // 获取当前设备支持的相机device列表
      this.cameras = this.cameraManager.getSupportedCameras();
      if (!this.cameras) {
        console.error('initCamera getSupportedCameras');
      }
      // 选择一个相机device，创建cameraInput输出对象
      this.cameraInput = this.cameraManager.createCameraInput(this.cameras[this.cameraID]);
      if (!this.cameraInput) {
        console.error('initCamera createCameraInput');
      }
      // 打开相机
      await this.cameraInput.open()
        .catch((err: BusinessError) => {
          console.error(`initCamera open fail: ${JSON.stringify(err)}`);
        })
      // 获取相机device支持的profile
      let capability: camera.CameraOutputCapability =
        this.cameraManager.getSupportedOutputCapability(this.cameras[this.cameraID], camera.SceneMode.NORMAL_VIDEO);
      if (!capability) {
        console.error('initCamera getSupportedOutputCapability');
      }

      let delta: number = Number.MAX_VALUE
      let profileIndex: number = 0
      if (capability.previewProfiles.length > 0) {
        for (let i = 0; i < capability.previewProfiles.length; i++) {
          const temp: camera.Profile = capability.previewProfiles[i]
          console.log(`profile ${i}: ${temp.size.width}x${temp.size.height}`)
          let deltaTemp: number = Math.abs(temp.size.width * temp.size.height - this.imageWidth * this.imageHeight)
          if (deltaTemp < delta) {
            delta = deltaTemp
            profileIndex = i
          }
        }
      }

      // 根据业务需求选择一个支持的预览流profile
      let previewProfile: camera.Profile = capability.previewProfiles[profileIndex];
      this.imageWidth = previewProfile.size.width; // 更新xComponent组件的宽
      this.imageHeight = previewProfile.size.height; // 更新xComponent组件的高
      console.info(`initCamera imageWidth:${this.imageWidth} imageHeight:${this.imageHeight}`);
      // 使用imageReceiverSurfaceId创建第一路预览
      this.previewOutput1 = this.cameraManager.createPreviewOutput(previewProfile, this.imageReceiverSurfaceId);
      if (!this.previewOutput1) {
        console.error('initCamera createPreviewOutput1');
      }
      // 使用xComponentSurfaceId创建第二路预览
      this.previewOutput2 = this.cameraManager.createPreviewOutput(previewProfile, this.xComponentSurfaceId);
      if (!this.previewOutput2) {
        console.error('initCamera createPreviewOutput2');
      }
      // 创建录像模式相机会话
      this.session = this.cameraManager.createSession(camera.SceneMode.NORMAL_VIDEO) as camera.VideoSession;
      if (!this.session) {
        console.error('initCamera createSession');
      }
      // 开始配置会话
      this.session.beginConfig();
      // 添加相机设备输入
      this.session.addInput(this.cameraInput);
      // 添加第一路预览流输出
      this.session.addOutput(this.previewOutput1);
      // 添加第二路预览流输出
      this.session.addOutput(this.previewOutput2);
      // 提交会话配置
      await this.session.commitConfig();
      // 开始启动已配置的输入输出流
      await this.session.start();
    } catch (error) {
      console.error(`initCamera fail: ${JSON.stringify(error)}`);
    }
  }

  // 释放相机
  async releaseCamera(): Promise<void> {
    console.info('releaseCamera E');
    try {
      // 停止当前会话
      await this.session?.stop();
      // 释放相机输入流
      await this.cameraInput?.close();
      // 释放预览输出流
      await this.previewOutput1?.release();
      // 释放拍照输出流
      await this.previewOutput2?.release();
      // 释放会话
      await this.session?.release();
    } catch (error) {
      console.error(`releaseCamera fail: ${JSON.stringify(error)}`);
    }
  }
}
