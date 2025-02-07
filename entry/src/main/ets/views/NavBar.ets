import { TOP_HEIGHT } from "../utils/ConstantKey"
import { router } from "@kit.ArkUI"

@Component
export struct NavBar {
  title: string | Resource = '标题'
  showBack: boolean = true
  // true:正常返回，false:阻止返回
  onClickBack: () => boolean = () => {
    return true
  }
  @StorageLink(TOP_HEIGHT) topHeight: number = AppStorage.get(TOP_HEIGHT) || 0
  @BuilderParam leftBuilder: () => void = this.defaultLeftBuilder
  @BuilderParam centerBuilder: () => void = this.defaultCenterBuilder
  @BuilderParam rightBuilder: () => void = this.defaultRightBuilder

  // 左侧
  @Builder
  defaultLeftBuilder() {
    Image($r("app.media.icon_arrow_left_white"))
      .width($r('app.integer.navbar_height'))
      .height($r('app.integer.navbar_height'))
      .padding(12)
      .objectFit(ImageFit.Contain)
      .onClick(() => {
        if (this.onClickBack) {
          if (this.onClickBack()) {
            router.back()
          }
        } else {
          router.back()
        }
      })
  }

  // 中间
  @Builder
  defaultCenterBuilder() {
    Text(this.title)
      .width('100%')
      .textAlign(TextAlign.Center)
      .fontColor($r("sys.color.white"))// 都用白色
      .fontSize(18)
  }

  // 右侧
  @Builder
  defaultRightBuilder() {
    // Text('右侧')
    //   .fontColor($r("sys.color.white"))// 都用白色
    //   .padding(12)
  }

  aboutToAppear(): void {

  }

  build() {
    Column() {
      Row() {
        Row() {
          if (this.showBack) {
            this.leftBuilder()
          }
        }
        .justifyContent(FlexAlign.Start)
        .layoutWeight(1)

        Row() {
          this.centerBuilder()
        }
        .justifyContent(FlexAlign.Center)
        .layoutWeight(2)

        Row() {
          this.rightBuilder()
        }
        .justifyContent(FlexAlign.End)
        .layoutWeight(1)

      }
      .width('100%')
      .height($r('app.integer.navbar_height'))
    }
    .backgroundColor($r("app.color.navbar_color"))
    .padding({ top: this.topHeight })
  }
}
