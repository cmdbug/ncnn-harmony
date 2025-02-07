import tncnn from 'libtncnn.so'

@Entry
@ComponentV2
struct Tncnn {
  @Local message: string = 'ncnn'
  @Local version: string = tncnn.ncnn_version()

  build() {
    Row() {
      Column({ space: 5 }) {
        Image($r('app.media.ncnn_icon'))
          .width(45)
          .height(45)
          .opacity(0.5)
        Text(this.message + ': ' + this.version)
      }
      .width('100%')
    }
    .height('100%')
  }

}
