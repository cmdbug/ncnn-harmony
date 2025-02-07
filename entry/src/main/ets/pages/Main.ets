import { BOTTOM_HEIGHT, TOP_HEIGHT } from "../utils/ConstantKey"
import { promptAction, router } from "@kit.ArkUI"
import tncnn from 'libtncnn.so'
import resourceManager from '@ohos.resourceManager';
import { PermissionUtil } from '@pura/harmony-utils';


@Entry
@ComponentV2
struct Index {
  @Local topHeight: number = AppStorage.get(TOP_HEIGHT) || 0
  @Local bottomHeight: number = AppStorage.get(BOTTOM_HEIGHT) || 0
  @Local ncnnVersion: string = '1.0.00000000'
  resMgr: resourceManager.ResourceManager | undefined

  handlerClickPhoto() {
    router.pushUrl({
      url: 'pages/PhotoPage'
    })
  }

  handlerClickCamera() {
    PermissionUtil.checkRequestPermissions("ohos.permission.CAMERA")
      .then((fulfilled: boolean) => {
        if (fulfilled) {
          router.pushUrl({
            url: 'pages/CameraPage'
          })
        } else {
          promptAction.showToast({ message: $r('app.string.permission_camera') })
        }
      }, () => {

      })
      .catch(() => {

      })
      .finally(() => {

      })

  }

  aboutToAppear(): void {
    try {
      this.ncnnVersion = tncnn.ncnn_version()
    } catch (e) {
      console.log('获取版本异常:' + e.toString())
    }
  }

  build() {
    Column() {
      Image($r('app.media.ohhh'))
        .width('100%')
        .aspectRatio(1.67)
        .padding({ top: this.topHeight })
        .backgroundColor('#031829')

      List({ space: 7 }) {
        ListItem() {
          Button('图片')
            .width('100%')
            .fontColor($r('sys.color.white'))
            .backgroundColor($r('app.color.color_primary'))
            .onClick(() => {
              this.handlerClickPhoto()
            })
        }

        ListItem() {
          Button('摄像头')
            .width('100%')
            .fontColor($r('sys.color.white'))
            .backgroundColor($r('app.color.color_primary'))
            .onClick(() => {
              this.handlerClickCamera()
            })
        }
      }
      .width('100%')
      .height('100%')
      .layoutWeight(1)
      .padding({
        top: 7,
        left: 7,
        right: 7,
        bottom: this.bottomHeight
      })

      Column({ space: 5 }) {
        Image($r('app.media.ncnn_icon'))
          .width(45)
          .height(45)
          .opacity(0.5)
        Text(this.ncnnVersion)
          .fontSize(14)
          .fontColor('#999999')
      }
      .padding({ bottom: this.bottomHeight })

    }
    .height('100%')
    .width('100%')
    .backgroundColor($r('app.color.page_background_color'))
  }
}
